from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.config import dataclass_from_section
from utils.dataloader import PATCH_SIZE, SOURCE_SIZE, create_dataloaders
from utils.metrics import compute_binary_mask_metrics, get_postprocessed_mask
from utils.pretrained_unet import get_model


@dataclass(frozen=True)
class VisualizeConfig:
    dataset: str = "dataset_tu"
    base_dir: str = "."
    k_folds: int = 5
    batch_size: int = 1
    threshold: float = 0.5
    use_amp: bool = True
    model_root: str = "model"
    experiment_name: str | None = None
    required_conda_env: str = ""
    max_images_per_fold: int | None = None


# Edit these values to evaluate an experiment without passing CLI arguments.
CONFIG = VisualizeConfig(
    dataset="dataset_tu",
    base_dir=".",
    k_folds=5,
    batch_size=1,
    threshold=0.5,
    use_amp=True,
    model_root="model",
    experiment_name=None,
    required_conda_env="",
    max_images_per_fold=None,
)


def load_visualize_config(config_path: str | Path) -> VisualizeConfig:
    config = dataclass_from_section(CONFIG, config_path, "visualize")
    if config.experiment_name == "":
        config = VisualizeConfig(**{**asdict(config), "experiment_name": None})
    return config


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/default.toml")
    config_args, remaining_args = pre_parser.parse_known_args()
    config = load_visualize_config(config_args.config)

    parser = argparse.ArgumentParser(description="Evaluate trained folds and save prediction visualizations.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--dataset", default=config.dataset, choices=["tu", "drac", "dataset_tu", "dataset_drac"])
    parser.add_argument("--base-dir", default=config.base_dir)
    parser.add_argument("--k-folds", type=int, default=config.k_folds)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--threshold", type=float, default=config.threshold)
    parser.add_argument("--amp", "--use-amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=config.use_amp)
    parser.add_argument("--model-root", default=config.model_root)
    parser.add_argument("--experiment-name", default=config.experiment_name)
    parser.add_argument("--required-conda-env", default=config.required_conda_env)
    parser.add_argument("--max-images-per-fold", type=int, default=config.max_images_per_fold)
    args = parser.parse_args(remaining_args)
    if args.experiment_name == "":
        args.experiment_name = None
    return args


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
            f"This script expects conda environment '{required_env}', "
            f"but active environment is '{active_env}'. Run: conda activate {required_env}"
        )


def _format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_experiment_name(args: argparse.Namespace) -> str:
    dataset_name = str(args.dataset).replace("dataset_", "")
    amp_name = "amp" if args.use_amp else "fp32"
    return f"{dataset_name}_raw_with_augment_augx10_folds{args.k_folds}_ep100_bs8_lr0p0001_{amp_name}_plateau_unet_efficientnetb3"


def get_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        return output["out"]
    return output


def load_experiment_config(experiment_dir: Path) -> dict[str, object]:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_args_from_config(args: argparse.Namespace, experiment_dir: Path) -> argparse.Namespace:
    saved_config = load_experiment_config(experiment_dir)
    for key in ("dataset", "base_dir", "k_folds", "batch_size", "threshold"):
        if key in saved_config:
            setattr(args, key, saved_config[key])
    return args


def stitch_batch(
    image: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    patch_id: int,
    full_input: torch.Tensor,
    full_mask: torch.Tensor,
    full_logits: torch.Tensor,
) -> None:
    y = 0 if patch_id < 2 else PATCH_SIZE
    x = 0 if patch_id % 2 == 0 else PATCH_SIZE

    full_input[:, :, y : y + PATCH_SIZE, x : x + PATCH_SIZE] = image
    full_mask[:, :, y : y + PATCH_SIZE, x : x + PATCH_SIZE] = mask
    full_logits[:, :, y : y + PATCH_SIZE, x : x + PATCH_SIZE] = logits


def make_confusion_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~target_bool).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def save_visualization(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    save_path: Path,
    title: str,
) -> None:
    image_np = image[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    error_map = np.zeros_like(target_np)
    error_map[(pred_np == 1) & (target_np == 1)] = 1
    error_map[(pred_np == 1) & (target_np == 0)] = 2
    error_map[(pred_np == 0) & (target_np == 1)] = 3
    error_cmap = ListedColormap(["black", "lime", "red", "blue"])

    counts = make_confusion_counts(pred_np, target_np)
    cm_data = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle(title)

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(target_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(error_map, cmap=error_cmap, vmin=0, vmax=3)
    axes[3].set_title("Error Map (G=TP, R=FP, B=FN)")
    axes[3].axis("off")

    axes[4].imshow(cm_data, cmap="Greens", interpolation="nearest", vmin=0)
    axes[4].set_title("Confusion Matrix")
    axes[4].set_xticks([0, 1])
    axes[4].set_xticklabels(["0 Bg", "1 Vs"])
    axes[4].set_yticks([0, 1])
    axes[4].set_yticklabels(["0 Bg", "1 Vs"])
    axes[4].set_xlabel("Predicted")
    axes[4].set_ylabel("Ground Truth")

    threshold = cm_data.max() / 2.0 if cm_data.max() else 0
    for row, col in np.ndindex(cm_data.shape):
        color = "white" if cm_data[row, col] > threshold else "black"
        axes[4].text(col, row, f"{cm_data[row, col]:,}", ha="center", va="center", color=color, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def evaluate_fold(
    fold: int,
    args: argparse.Namespace,
    experiment_dir: Path,
    device: torch.device,
) -> dict[str, float] | None:
    weights_path = experiment_dir / "weights" / f"{fold}.pth"
    if not weights_path.exists():
        print(f"Missing weights for fold {fold}: {weights_path}")
        return None

    loaders = create_dataloaders(
        dataset=args.dataset,
        augment=False,
        aug_multiplier=1,
        fold=fold,
        batch_size=args.batch_size,
        base_dir=args.base_dir,
        num_workers=0,
        include_metadata=True,
        pin_memory=device.type == "cuda",
    )
    test_loader = loaders["test"]

    model = get_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    fold_output_dir = experiment_dir / "outputs" / f"test_fold_{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    image_metrics: list[dict[str, float]] = []
    buffers: dict[int, dict[str, torch.Tensor | set[int]]] = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold} [Test]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(args.use_amp and device.type == "cuda")):
                logits = get_logits(model, images)
            sample_ids = batch["sample_id"]
            patch_ids = batch["patch_id"]

            for item_index, sample_id in enumerate(sample_ids):
                sample_id_int = int(sample_id)
                if sample_id_int not in buffers:
                    buffers[sample_id_int] = {
                        "input": torch.zeros((1, 1, SOURCE_SIZE, SOURCE_SIZE)),
                        "mask": torch.zeros((1, 1, SOURCE_SIZE, SOURCE_SIZE)),
                        "logits": torch.zeros((1, 1, SOURCE_SIZE, SOURCE_SIZE)),
                        "patches": set(),
                    }

                buffer = buffers[sample_id_int]
                full_input = buffer["input"]
                full_mask = buffer["mask"]
                full_logits = buffer["logits"]
                patches = buffer["patches"]
                assert isinstance(full_input, torch.Tensor)
                assert isinstance(full_mask, torch.Tensor)
                assert isinstance(full_logits, torch.Tensor)
                assert isinstance(patches, set)

                patch_id = int(patch_ids[item_index])
                stitch_batch(
                    images[item_index : item_index + 1].detach().cpu(),
                    masks[item_index : item_index + 1].detach().cpu(),
                    logits[item_index : item_index + 1].float().detach().cpu(),
                    patch_id,
                    full_input,
                    full_mask,
                    full_logits,
                )
                patches.add(patch_id)

                if len(patches) == 4:
                    pred = get_postprocessed_mask(full_logits, threshold=args.threshold)
                    metrics = compute_binary_mask_metrics(pred, full_mask)
                    metrics["sample_id"] = float(sample_id_int)
                    image_metrics.append(metrics)

                    save_visualization(
                        full_input,
                        full_mask,
                        pred,
                        fold_output_dir / f"sample_{sample_id_int}.png",
                        title=f"Fold {fold} Sample {sample_id_int}",
                    )

                    del buffers[sample_id_int]

                    if args.max_images_per_fold and len(image_metrics) >= args.max_images_per_fold:
                        break

            if args.max_images_per_fold and len(image_metrics) >= args.max_images_per_fold:
                break

    if not image_metrics:
        return None

    fold_summary = {
        "fold": float(fold),
        "dice": float(np.mean([item["dice"] for item in image_metrics])),
        "iou": float(np.mean([item["iou"] for item in image_metrics])),
        "precision": float(np.mean([item["precision"] for item in image_metrics])),
        "recall": float(np.mean([item["recall"] for item in image_metrics])),
        "specificity": float(np.mean([item["specificity"] for item in image_metrics])),
        "accuracy": float(np.mean([item["accuracy"] for item in image_metrics])),
        "num_images": float(len(image_metrics)),
    }

    per_image_path = fold_output_dir / "per_image_metrics.csv"
    write_csv(per_image_path, image_metrics)
    return fold_summary


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_conda_env(args.required_conda_env)

    experiment_name = args.experiment_name or build_experiment_name(args)
    experiment_dir = Path(args.model_root) / experiment_name
    args = resolve_args_from_config(args, experiment_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Device: {device}")

    summaries: list[dict[str, float]] = []
    for fold in range(1, int(args.k_folds) + 1):
        summary = evaluate_fold(fold, args, experiment_dir, device)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No fold metrics were produced. Check experiment name and weight files.")

    outputs_dir = experiment_dir / "outputs"
    write_csv(outputs_dir / "test_metrics.csv", summaries)
    (outputs_dir / "test_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (outputs_dir / "visualize_config.json").write_text(
        json.dumps({**asdict(load_visualize_config(args.config)), **vars(args), "experiment_name": experiment_name}, indent=2),
        encoding="utf-8",
    )

    print("\nTest metrics:")
    for summary in summaries:
        print(
            f"Fold {int(summary['fold'])}: "
            f"Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, "
            f"Precision={summary['precision']:.4f}, Recall={summary['recall']:.4f}"
        )
    print(f"\nSaved visualizations and metrics under: {outputs_dir}")


if __name__ == "__main__":
    main()
