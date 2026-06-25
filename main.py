from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import random
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm.auto import tqdm

from utils.config import dataclass_from_section
from utils.dataloader import create_dataloaders
from utils.metrics import BCEDiceLoss, compute_binary_metrics, dice_score_from_logits
from utils.model_factory import get_model, model_experiment_token, normalize_backbone_name, normalize_model_name


MASK_METRICS = ("dice", "iou", "precision", "recall", "specificity", "accuracy")
BEST_MODEL_METRICS = ("loss", *MASK_METRICS)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "dataset_tu"
    model_name: str = "unet"
    backbone: str = "efficientnet-b3"
    augment: bool = True
    aug_multiplier: int = 10
    base_dir: str = "."
    k_folds: int = 5
    epochs: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 1e-4
    use_amp: bool = True
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-7
    patience: int = 30
    num_workers: int = 0
    threshold: float = 0.5
    best_metric: str = "dice"
    seed: int = 42
    model_root: str = "model"
    required_conda_env: str = ""
    experiment_name: str | None = None


# Edit these values for a new experiment. CLI arguments can still override them.
CONFIG = ExperimentConfig(
    dataset="dataset_drac",
    model_name="unet",
    backbone="efficientnet-b3",
    augment=True,
    aug_multiplier=10,
    base_dir=".",
    k_folds=5,
    epochs=100,
    batch_size=8,
    grad_accum_steps=1,
    lr=1e-4,
    use_amp=True,
    use_scheduler=True,
    scheduler_factor=0.5,
    scheduler_patience=3,
    min_lr=1e-7,
    patience=30,
    num_workers=0,
    threshold=0.5,
    best_metric="dice",
    seed=42,
    model_root="model",
    required_conda_env="",
    experiment_name=None,
)


def load_train_config(config_path: str | Path) -> ExperimentConfig:
    config = dataclass_from_section(CONFIG, config_path, "train")
    if config.experiment_name == "":
        config = ExperimentConfig(**{**asdict(config), "experiment_name": None})
    return config


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/default.toml")
    config_args, remaining_args = pre_parser.parse_known_args()
    config = load_train_config(config_args.config)

    parser = argparse.ArgumentParser(description="Train a segmentation model with fixed k-fold splits.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--dataset", default=config.dataset, choices=["tu", "drac", "dataset_tu", "dataset_drac"])
    parser.add_argument(
        "--model-name",
        default=config.model_name,
        help="Model architecture: unet, unetplusplus, deeplabv3plus, or deeplabv3.",
    )
    parser.add_argument(
        "--backbone",
        default=config.backbone,
        help="Encoder backbone: efficientnet-b3, inceptionv4, or densenet169.",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=config.augment)
    parser.add_argument("--aug-multiplier", type=int, default=config.aug_multiplier)
    parser.add_argument("--base-dir", default=config.base_dir)
    parser.add_argument("--k-folds", type=int, default=config.k_folds)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=config.grad_accum_steps,
        help="Accumulate gradients across this many mini-batches before optimizer step.",
    )
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument(
        "--amp",
        "--use-amp",
        dest="use_amp",
        action=argparse.BooleanOptionalAction,
        default=config.use_amp,
    )
    parser.add_argument(
        "--scheduler",
        "--use-scheduler",
        dest="use_scheduler",
        action=argparse.BooleanOptionalAction,
        default=config.use_scheduler,
    )
    parser.add_argument("--scheduler-factor", type=float, default=config.scheduler_factor)
    parser.add_argument("--scheduler-patience", type=int, default=config.scheduler_patience)
    parser.add_argument("--min-lr", type=float, default=config.min_lr)
    parser.add_argument("--patience", type=int, default=config.patience)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--threshold", type=float, default=config.threshold)
    parser.add_argument(
        "--best-metric",
        default=config.best_metric,
        help=(
            "Validation metric used for saving the best model and early stopping. "
            f"Choices: {', '.join(BEST_MODEL_METRICS)}. Prefix 'val_' is also accepted."
        ),
    )
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--model-root", default=config.model_root)
    parser.add_argument("--required-conda-env", default=config.required_conda_env)
    parser.add_argument("--experiment-name", default=config.experiment_name)
    args = parser.parse_args(remaining_args)
    if args.experiment_name == "":
        args.experiment_name = None
    try:
        args.model_name = normalize_model_name(args.model_name)
        args.backbone = normalize_backbone_name(args.backbone)
        args.best_metric = normalize_best_metric(args.best_metric)
    except ValueError as exc:
        parser.error(str(exc))
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be at least 1")
    return args


def normalize_best_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized.startswith("val_"):
        normalized = normalized[4:]

    if normalized not in BEST_MODEL_METRICS:
        choices = ", ".join(BEST_MODEL_METRICS)
        raise ValueError(f"Invalid --best-metric '{metric}'. Expected one of: {choices}")
    return normalized


def best_metric_mode(metric: str) -> str:
    return "min" if metric == "loss" else "max"


def is_better_metric_value(value: float, best_value: float | None, metric: str) -> bool:
    if best_value is None:
        return True
    if best_metric_mode(metric) == "min":
        return value < best_value
    return value > best_value


def format_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_conda_env_name() -> str | None:
    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_default_env:
        return conda_default_env

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix).name

    executable_parts = Path(sys.executable).parts
    if "envs" in executable_parts:
        envs_index = executable_parts.index("envs")
        if envs_index + 1 < len(executable_parts):
            return executable_parts[envs_index + 1]
    return None


def ensure_conda_env(required_env: str | None) -> None:
    if not required_env:
        return

    active_env = get_conda_env_name()
    if active_env != required_env:
        raise RuntimeError(
            f"This experiment expects conda environment '{required_env}', "
            f"but active environment is '{active_env}'. Run: conda activate {required_env}"
        )


def _format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_experiment_name(args: argparse.Namespace) -> str:
    aug_name = "with_augment" if args.augment else "no_augment"
    scheduler_name = "plateau" if args.use_scheduler else "no_scheduler"
    dataset_name = str(args.dataset).replace("dataset_", "")
    amp_name = "amp" if args.use_amp else "fp32"
    model_token = model_experiment_token(args.model_name, args.backbone)
    return (
        f"{dataset_name}_raw_{aug_name}"
        f"_augx{args.aug_multiplier}"
        f"_folds{args.k_folds}"
        f"_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_ga{args.grad_accum_steps}"
        f"_lr{_format_float(args.lr)}"
        f"_{amp_name}"
        f"_{scheduler_name}"
        f"_{model_token}"
    )


def get_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        if "main_out" in output:
            return output["main_out"]
        if len(output) == 1:
            return next(iter(output.values()))
        keys = ", ".join(sorted(output))
        raise KeyError(f"Model output must contain 'out' logits. Available keys: {keys}")
    return output


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if not args.use_scheduler:
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=best_metric_mode(args.best_metric),
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    dices: list[float] = []

    loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for batch_index, batch in enumerate(loop, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = get_logits(model, images)
            loss = criterion(logits, masks)

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        if batch_index % grad_accum_steps == 0 or batch_index == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu().item()))
        dice = dice_score_from_logits(logits.detach(), masks.detach(), threshold=threshold)
        dices.append(float(dice.detach().cpu().item()))
        loop.set_postfix(loss=losses[-1], dice=dices[-1], accum=grad_accum_steps)

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_values: dict[str, list[float]] = {name: [] for name in MASK_METRICS}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = get_logits(model, images)
            loss = criterion(logits, masks)

        metrics = compute_binary_metrics(logits.float(), masks, threshold=threshold, postprocess=True)

        losses.append(float(loss.detach().cpu().item()))
        for name in MASK_METRICS:
            metric_values[name].append(metrics[name])

    return {
        "loss": float(np.mean(losses)),
        **{name: float(np.mean(values)) for name, values in metric_values.items()},
    }


def save_history_plot(history: dict[str, list[float]], output_dir: Path, experiment_name: str, fold: int) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["val_loss"], label="Validation Loss", marker="x")
    ax1.set_title(f"Loss Curve - {experiment_name} (Fold {fold})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_dice"], label="Train Dice", marker="o")
    ax2.plot(history["val_dice"], label="Validation Dice", marker="x")
    ax2.set_title(f"Dice Curve - {experiment_name} (Fold {fold})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(history["lr"], label="Learning Rate", marker="o")
    ax3.set_title(f"LR Schedule - {experiment_name} (Fold {fold})")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"loss_dice_fold_{fold}.png"
    fig.savefig(graph_path, bbox_inches="tight")
    plt.close(fig)


def save_history_json(history: dict[str, list[float]], output_dir: Path, fold: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"history_fold_{fold}.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_best_model_timing_json(
    output_dir: Path,
    fold: int,
    timing: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / f"best_model_timing_fold_{fold}.json"
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")


def format_elapsed_time(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "NA"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def describe_loader_samples(loaders: dict[str, torch.utils.data.DataLoader]) -> None:
    train_dataset = loaders["train"].dataset
    validation_dataset = loaders["validation"].dataset
    test_dataset = loaders["test"].dataset

    train_base = len(getattr(train_dataset, "samples", train_dataset))
    train_after_augmentation = len(train_dataset)
    validation_count = len(validation_dataset)
    test_count = len(test_dataset)

    print("Split sample counts:")
    print(f"  Train base images: {train_base}")
    print(f"  Train after augmentation: {train_after_augmentation}")
    print(f"  Validation images: {validation_count}")
    print(f"  Test images: {test_count}")


def cuda_oom_message(args: argparse.Namespace, fold: int) -> str:
    return (
        f"CUDA out of memory during fold {fold}. "
        "The pipeline now trains on full 1024x1024 images. "
        "Try lowering --batch-size, keeping --amp enabled, or raising "
        "--grad-accum-steps to preserve effective batch size while reducing mini-batch memory."
    )


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    weights_dir: Path,
    output_dir: Path,
    experiment_name: str,
) -> dict[str, Any]:
    print(f"\n{'=' * 40}")
    print(f"Starting fold {fold}/{args.k_folds}")
    print(f"{'=' * 40}")

    loaders = create_dataloaders(
        dataset=args.dataset,
        augment=args.augment,
        aug_multiplier=args.aug_multiplier,
        fold=fold,
        batch_size=args.batch_size,
        base_dir=args.base_dir,
        num_workers=args.num_workers,
        include_metadata=False,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )
    describe_loader_samples(loaders)

    print(
        "Batching: "
        f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, "
        f"effective_train_batch={args.batch_size * args.grad_accum_steps}"
    )

    model = get_model(
        device,
        model_name=args.model_name,
        backbone=args.backbone,
        in_channels=1,
        out_channels=1,
    )
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args)
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_dice": [],
        **{f"val_{name}": [] for name in BEST_MODEL_METRICS},
        "lr": [],
        "epoch_elapsed_seconds": [],
        "best_model_saved_elapsed_seconds": [],
    }

    best_metric_value: float | None = None
    best_model_saved_epoch: int | None = None
    best_model_saved_elapsed_seconds: float | None = None
    patience_counter = 0
    best_model_path = weights_dir / f"{fold}.pth"
    best_metric_label = f"Val {format_metric_name(args.best_metric)}"
    print(
        f"Best model metric: {best_metric_label} "
        f"({best_metric_mode(args.best_metric)}, threshold={args.threshold:g})"
    )

    fold_train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        try:
            train_metrics = train_one_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                threshold=args.threshold,
                scaler=scaler,
                use_amp=use_amp,
                grad_accum_steps=args.grad_accum_steps,
                epoch=epoch,
                epochs=args.epochs,
            )
            val_metrics = validate(
                model=model,
                loader=loaders["validation"],
                criterion=criterion,
                device=device,
                threshold=args.threshold,
                use_amp=use_amp,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(cuda_oom_message(args, fold)) from exc

        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        for name in BEST_MODEL_METRICS:
            history[f"val_{name}"].append(val_metrics[name])

        previous_lr = get_current_lr(optimizer)
        if scheduler is not None:
            scheduler.step(val_metrics[args.best_metric])
        current_lr = get_current_lr(optimizer)
        history["lr"].append(current_lr)
        epoch_elapsed_seconds = time.perf_counter() - fold_train_start
        history["epoch_elapsed_seconds"].append(epoch_elapsed_seconds)

        log_message = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        if args.best_metric not in {"loss", "dice", "iou"}:
            log_message += f" | {best_metric_label}: {val_metrics[args.best_metric]:.4f}"
        if current_lr < previous_lr:
            log_message += f" (reduced from {previous_lr:.2e})"
        print(log_message)

        current_metric_value = val_metrics[args.best_metric]
        if is_better_metric_value(current_metric_value, best_metric_value, args.best_metric):
            best_metric_value = current_metric_value
            best_model_saved_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            best_model_saved_elapsed_seconds = time.perf_counter() - fold_train_start
            history["best_model_saved_elapsed_seconds"].append(best_model_saved_elapsed_seconds)
            save_best_model_timing_json(
                output_dir,
                fold,
                {
                    "fold": fold,
                    "epoch": epoch,
                    "best_metric": args.best_metric,
                    "best_metric_mode": best_metric_mode(args.best_metric),
                    f"best_val_{args.best_metric}": float(best_metric_value),
                    "best_model_path": str(best_model_path),
                    "best_model_saved_elapsed_seconds": best_model_saved_elapsed_seconds,
                    "best_model_saved_elapsed_hms": format_elapsed_time(best_model_saved_elapsed_seconds),
                },
            )
            print(
                f"Saved new best model: {best_model_path} "
                f"({best_metric_label}: {best_metric_value:.4f}, "
                f"elapsed: {format_elapsed_time(best_model_saved_elapsed_seconds)})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {args.patience} epochs without {best_metric_label} improvement.")
                break

    save_history_plot(history, output_dir, experiment_name, fold)
    save_history_json(history, output_dir, fold)

    return {
        "fold": float(fold),
        "best_metric": args.best_metric,
        "best_metric_mode": best_metric_mode(args.best_metric),
        f"best_val_{args.best_metric}": float(best_metric_value if best_metric_value is not None else np.nan),
        "best_model_saved_epoch": float(best_model_saved_epoch if best_model_saved_epoch is not None else np.nan),
        "best_model_saved_elapsed_seconds": float(
            best_model_saved_elapsed_seconds if best_model_saved_elapsed_seconds is not None else np.nan
        ),
        "best_model_saved_elapsed_hms": format_elapsed_time(best_model_saved_elapsed_seconds),
        **{f"last_val_{name}": history[f"val_{name}"][-1] for name in BEST_MODEL_METRICS},
    }


def main() -> None:
    args = parse_args()
    ensure_conda_env(args.required_conda_env)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = args.experiment_name or build_experiment_name(args)
    experiment_dir = Path(args.model_root) / experiment_name
    weights_dir = experiment_dir / "weights"
    output_dir = experiment_dir / "outputs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Model: {args.model_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Weights directory: {weights_dir}")
    print(f"Output directory: {output_dir}")

    config_path = experiment_dir / "config.json"
    config_data = asdict(load_train_config(args.config))
    config_data.update(vars(args))
    config_data["experiment_name"] = experiment_name
    config_data["active_conda_env"] = get_conda_env_name()
    config_data["python_executable"] = sys.executable
    config_data["conda_prefix"] = os.environ.get("CONDA_PREFIX")
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    fold_summaries = []
    for fold in range(1, args.k_folds + 1):
        fold_summaries.append(
            run_fold(
                fold=fold,
                args=args,
                device=device,
                weights_dir=weights_dir,
                output_dir=output_dir,
                experiment_name=experiment_name,
            )
        )

    summary_path = output_dir / "kfold_summary.json"
    summary_path.write_text(json.dumps(fold_summaries, indent=2), encoding="utf-8")
    print(f"Saved k-fold summary: {summary_path}")


if __name__ == "__main__":
    main()
