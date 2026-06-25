from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, label
from tqdm.auto import tqdm

from utils.config import dataclass_from_section
from utils.dataloader import create_dataloaders
from utils.metrics import compute_binary_mask_metrics, get_postprocessed_mask
from utils.model_factory import get_model, model_experiment_token, normalize_backbone_name, normalize_model_name


METRIC_COLUMNS = ("dice", "iou", "precision", "recall", "specificity", "accuracy")
DATASET_PREFIXES = ("tu", "drac")
STANDARD_OUTPUT_DIR = "outputs"
NEARBY_OUTPUT_DIR = "outputs_nearby"


@dataclass(frozen=True)
class VisualizeConfig:
    dataset: str = "dataset_tu"
    model_name: str = "unet"
    backbone: str = "efficientnet-b3"
    base_dir: str = "."
    k_folds: int = 5
    batch_size: int = 1
    threshold: float = 0.5
    use_amp: bool = True
    model_root: str = "model"
    experiment_name: str | None = None
    required_conda_env: str = ""
    max_images_per_fold: int | None = None
    summarize_all_datasets: bool = False
    summary_only: bool = False
    setting_suffix: str | None = None
    apply_nearby_filter: bool = False
    points_csv: str = ""
    points_image_name_template: str = "nv ({sample_id}).jpg"
    radius_tolerance: float = 25.0


CONFIG = VisualizeConfig(
    dataset="dataset_tu",
    model_name="unet",
    backbone="efficientnet-b3",
    base_dir=".",
    k_folds=5,
    batch_size=1,
    threshold=0.5,
    use_amp=True,
    model_root="model",
    experiment_name=None,
    required_conda_env="",
    max_images_per_fold=None,
    summarize_all_datasets=False,
    summary_only=False,
    setting_suffix=None,
    apply_nearby_filter=False,
    points_csv="",
    points_image_name_template="nv ({sample_id}).jpg",
    radius_tolerance=25.0,
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
    parser.add_argument("--base-dir", default=config.base_dir)
    parser.add_argument("--k-folds", type=int, default=config.k_folds)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--threshold", type=float, default=config.threshold)
    parser.add_argument("--amp", "--use-amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=config.use_amp)
    parser.add_argument("--model-root", default=config.model_root)
    parser.add_argument("--experiment-name", default=config.experiment_name)
    parser.add_argument("--required-conda-env", default=config.required_conda_env)
    parser.add_argument("--max-images-per-fold", type=int, default=config.max_images_per_fold)
    parser.add_argument("--summarize-all-datasets", action=argparse.BooleanOptionalAction, default=config.summarize_all_datasets)
    parser.add_argument("--summary-only", action=argparse.BooleanOptionalAction, default=config.summary_only)
    parser.add_argument("--setting-suffix", default=config.setting_suffix)
    parser.add_argument(
        "--nearby-filter",
        "--apply-nearby-filter",
        dest="apply_nearby_filter",
        action=argparse.BooleanOptionalAction,
        default=config.apply_nearby_filter,
        help="Keep predicted components that are within radius_tolerance pixels of a representative point.",
    )
    parser.add_argument("--points-csv", default=config.points_csv)
    parser.add_argument("--points-image-name-template", default=config.points_image_name_template)
    parser.add_argument("--radius-tolerance", type=float, default=config.radius_tolerance)
    args = parser.parse_args(remaining_args)
    if args.experiment_name == "":
        args.experiment_name = None
    if args.setting_suffix == "":
        args.setting_suffix = None
    if args.points_csv == "":
        args.points_csv = None
    try:
        args.model_name = normalize_model_name(args.model_name)
        args.backbone = normalize_backbone_name(args.backbone)
    except ValueError as exc:
        parser.error(str(exc))
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
    model_token = model_experiment_token(args.model_name, args.backbone)
    return (
        f"{dataset_name}_raw_with_augment_augx10_folds{args.k_folds}_ep100_bs8_lr0p0001_"
        f"{amp_name}_plateau_{model_token}"
    )


def get_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        if len(output) == 1:
            return next(iter(output.values()))
        keys = ", ".join(sorted(output))
        raise KeyError(f"Model output must contain 'out' logits. Available keys: {keys}")
    return output


def load_experiment_config(experiment_dir: Path) -> dict[str, object]:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_args_from_config(args: argparse.Namespace, experiment_dir: Path) -> argparse.Namespace:
    saved_config = load_experiment_config(experiment_dir)
    for key in ("dataset", "model_name", "backbone", "base_dir", "k_folds", "batch_size", "threshold"):
        if key in saved_config:
            setattr(args, key, saved_config[key])
    args.model_name = normalize_model_name(args.model_name)
    args.backbone = normalize_backbone_name(getattr(args, "backbone", "efficientnet-b3"))
    return args


def get_output_subdir(apply_nearby_filter: bool) -> str:
    return NEARBY_OUTPUT_DIR if apply_nearby_filter else STANDARD_OUTPUT_DIR


def get_outputs_dir(experiment_dir: Path, apply_nearby_filter: bool) -> Path:
    return experiment_dir / get_output_subdir(apply_nearby_filter)


def make_confusion_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~target_bool).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def filter_by_representative_points(
    pred_mask: torch.Tensor,
    image_name: str,
    points_df: pd.DataFrame,
    radius_tolerance: float,
) -> tuple[torch.Tensor, np.ndarray | None]:
    """
    Keep predicted connected components that are within `radius_tolerance`
    pixels of at least one representative localization point.
    """
    pred_np = pred_mask[0, 0].detach().cpu().numpy().astype(np.uint8)

    img_points = points_df[points_df["image_number"] == image_name]
    if img_points.empty:
        return torch.zeros_like(pred_mask), None

    # CSV coordinates are expected as x=column and y=row.
    ref_points = img_points[["y", "x"]].to_numpy(dtype=np.float64)

    labeled_mask, num_features = label(pred_np)
    if num_features == 0:
        return pred_mask, ref_points

    rows = np.rint(ref_points[:, 0]).astype(np.int64)
    cols = np.rint(ref_points[:, 1]).astype(np.int64)
    rows = np.clip(rows, 0, labeled_mask.shape[0] - 1)
    cols = np.clip(cols, 0, labeled_mask.shape[1] - 1)

    points_mask = np.zeros_like(pred_np, dtype=bool)
    points_mask[rows, cols] = True
    dist_map = distance_transform_edt(~points_mask)
    within_radius_mask = dist_map <= radius_tolerance

    hit_labels = labeled_mask[within_radius_mask]
    valid_blob_ids = hit_labels[hit_labels > 0]
    unique_valid_blobs = np.unique(valid_blob_ids)

    clean_np = np.isin(labeled_mask, unique_valid_blobs).astype(np.uint8)
    cleaned_tensor = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(pred_mask.device, dtype=pred_mask.dtype)
    return cleaned_tensor, ref_points


def save_visualization(
    image: torch.Tensor,
    target: torch.Tensor,
    raw_pred: torch.Tensor,
    final_pred: torch.Tensor,
    ref_points: np.ndarray | None,
    save_path: Path,
    title: str,
    apply_nearby_filter: bool,
) -> None:
    image_np = image[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()
    raw_pred_np = raw_pred[0, 0].detach().cpu().numpy()
    final_pred_np = final_pred[0, 0].detach().cpu().numpy()

    error_map = np.zeros_like(target_np)
    error_map[(final_pred_np == 1) & (target_np == 1)] = 1
    error_map[(final_pred_np == 1) & (target_np == 0)] = 2
    error_map[(final_pred_np == 0) & (target_np == 1)] = 3
    error_cmap = ListedColormap(["black", "lime", "red", "blue"])

    counts = make_confusion_counts(final_pred_np, target_np)
    cm_data = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])

    if apply_nearby_filter:
        fig, axes = plt.subplots(1, 6, figsize=(28, 5))
    else:
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle(title)

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(target_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(raw_pred_np, cmap="gray", vmin=0, vmax=1)
    if ref_points is not None and len(ref_points) > 0:
        axes[2].scatter(
            ref_points[:, 1],
            ref_points[:, 0],
            c="red",
            s=60,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
        )
    axes[2].set_title("Raw Prediction" if apply_nearby_filter else "Prediction")
    axes[2].axis("off")

    if apply_nearby_filter:
        axes[3].imshow(final_pred_np, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Nearby Prediction")
        axes[3].axis("off")
        error_axis = axes[4]
        cm_axis = axes[5]
    else:
        error_axis = axes[3]
        cm_axis = axes[4]

    error_axis.imshow(error_map, cmap=error_cmap, vmin=0, vmax=3)
    error_axis.set_title("Error Map (G=TP, R=FP, B=FN)")
    error_axis.axis("off")

    cm_axis.imshow(cm_data, cmap="Greens", interpolation="nearest", vmin=0)
    cm_axis.set_title("Confusion Matrix")
    cm_axis.set_xticks([0, 1])
    cm_axis.set_xticklabels(["0 Bg", "1 Vs"])
    cm_axis.set_yticks([0, 1])
    cm_axis.set_yticklabels(["0 Bg", "1 Vs"])
    cm_axis.set_xlabel("Predicted")
    cm_axis.set_ylabel("Ground Truth")

    threshold = cm_data.max() / 2.0 if cm_data.max() else 0
    for row, col in np.ndindex(cm_data.shape):
        color = "white" if cm_data[row, col] > threshold else "black"
        cm_axis.text(col, row, f"{cm_data[row, col]:,}", ha="center", va="center", color=color, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def evaluate_fold(
    fold: int,
    args: argparse.Namespace,
    experiment_dir: Path,
    outputs_dir: Path,
    device: torch.device,
    points_df: pd.DataFrame | None,
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

    model = get_model(
        device,
        model_name=args.model_name,
        backbone=args.backbone,
        in_channels=1,
        out_channels=1,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    fold_output_dir = outputs_dir / f"test_fold_{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    image_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold} [Test]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(args.use_amp and device.type == "cuda")):
                logits = get_logits(model, images)
            if logits.shape[-2:] != images.shape[-2:]:
                raise ValueError(
                    f"Expected model output spatial size {tuple(images.shape[-2:])}, got {tuple(logits.shape[-2:])}"
                )
            sample_ids = batch["sample_id"]

            for item_index, sample_id in enumerate(sample_ids):
                sample_id_int = int(sample_id)
                full_input = images[item_index : item_index + 1].detach().cpu()
                full_mask = masks[item_index : item_index + 1].detach().cpu()
                full_logits = logits[item_index : item_index + 1].float().detach().cpu()
                raw_pred = get_postprocessed_mask(full_logits, threshold=args.threshold)

                final_pred = raw_pred
                ref_points = None
                if args.apply_nearby_filter:
                    assert points_df is not None
                    image_number = args.points_image_name_template.format(
                        sample_id=sample_id_int,
                        dataset=str(args.dataset).replace("dataset_", ""),
                    )
                    final_pred, ref_points = filter_by_representative_points(
                        pred_mask=raw_pred,
                        image_name=image_number,
                        points_df=points_df,
                        radius_tolerance=args.radius_tolerance,
                    )

                metrics = compute_binary_mask_metrics(final_pred, full_mask)
                metrics["sample_id"] = float(sample_id_int)
                metrics["num_reference_points"] = float(0 if ref_points is None else len(ref_points))
                image_metrics.append(metrics)

                save_visualization(
                    image=full_input,
                    target=full_mask,
                    raw_pred=raw_pred,
                    final_pred=final_pred,
                    ref_points=ref_points,
                    save_path=fold_output_dir / f"sample_{sample_id_int}.png",
                    title=f"Fold {fold} Sample {sample_id_int}",
                    apply_nearby_filter=args.apply_nearby_filter,
                )

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

    write_csv(fold_output_dir / "per_image_metrics.csv", image_metrics)
    return fold_summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {"mean": {}, "std": {}}
    for metric in METRIC_COLUMNS:
        values = [float(row[metric]) for row in rows if metric in row]
        if not values:
            continue
        summary["mean"][metric] = float(np.mean(values))
        summary["std"][metric] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary


def write_experiment_summary(outputs_dir: Path, fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_summary = summarize_metric_rows(fold_summaries)
    summary = {
        "num_folds": len(fold_summaries),
        "mean": metric_summary["mean"],
        "std": metric_summary["std"],
    }

    summary_rows: list[dict[str, Any]] = []
    for summary_type in ("mean", "std"):
        row: dict[str, Any] = {"summary": summary_type}
        row.update(metric_summary[summary_type])
        summary_rows.append(row)

    write_csv(outputs_dir / "test_metrics_summary.csv", summary_rows)
    (outputs_dir / "test_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def split_dataset_experiment_name(experiment_name: str) -> tuple[str, str] | None:
    for dataset in DATASET_PREFIXES:
        prefix = f"{dataset}_"
        if experiment_name.startswith(prefix):
            return dataset, experiment_name[len(prefix):]
    return None


def load_experiment_metric_summary(experiment_dir: Path, output_subdir: str = STANDARD_OUTPUT_DIR) -> dict[str, Any] | None:
    outputs_dir = experiment_dir / output_subdir
    summary_path = outputs_dir / "test_metrics_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    metrics_path = outputs_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None

    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not rows:
        return None
    return {
        "num_folds": len(rows),
        **summarize_metric_rows(rows),
    }


def load_experiment_metric_rows(experiment_dir: Path, output_subdir: str = STANDARD_OUTPUT_DIR) -> list[dict[str, Any]]:
    metrics_path = experiment_dir / output_subdir / "test_metrics.json"
    if not metrics_path.exists():
        return []

    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of fold metrics in {metrics_path}")
    return rows


def summarize_all_datasets(
    model_root: str | Path,
    setting_suffix: str,
    output_subdir: str = STANDARD_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    model_root = Path(model_root)
    rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []

    for dataset in DATASET_PREFIXES:
        experiment_name = f"{dataset}_{setting_suffix}"
        experiment_dir = model_root / experiment_name
        fold_rows = load_experiment_metric_rows(experiment_dir, output_subdir=output_subdir)
        summary = {
            "num_folds": len(fold_rows),
            **summarize_metric_rows(fold_rows),
        } if fold_rows else load_experiment_metric_summary(experiment_dir, output_subdir=output_subdir)
        if summary is None:
            print(f"Missing evaluated metrics for dataset '{dataset}': {experiment_dir / output_subdir}")
            continue
        if fold_rows:
            all_fold_rows.extend(fold_rows)
        else:
            print(f"Missing fold-level metrics for dataset '{dataset}': {experiment_dir / output_subdir / 'test_metrics.json'}")

        row: dict[str, Any] = {
            "dataset": dataset,
            "experiment_name": experiment_name,
            "num_folds": summary.get("num_folds", 0),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = summary.get("mean", {}).get(metric)
            row[f"{metric}_std"] = summary.get("std", {}).get(metric)
        rows.append(row)

    if not rows:
        return []

    if all_fold_rows:
        combined_summary = summarize_metric_rows(all_fold_rows)
        combined: dict[str, Any] = {
            "dataset": "all",
            "experiment_name": setting_suffix,
            "num_folds": float(len(all_fold_rows)),
        }
        for metric in METRIC_COLUMNS:
            combined[f"{metric}_mean"] = combined_summary["mean"].get(metric)
            combined[f"{metric}_std"] = combined_summary["std"].get(metric)
        rows.append(combined)
    else:
        print("Cannot compute all-fold combined metrics because no fold-level test_metrics.json files were found.")

    output_dir = model_root / "dataset_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = setting_suffix.replace("/", "_")
    output_name = safe_suffix if output_subdir == STANDARD_OUTPUT_DIR else f"{safe_suffix}_nearby"
    write_csv(output_dir / f"{output_name}.csv", rows)
    (output_dir / f"{output_name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def format_summary_metric(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None or std is None:
        return "NA"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def main() -> None:
    args = parse_args()
    ensure_conda_env(args.required_conda_env)

    experiment_name = args.experiment_name or build_experiment_name(args)
    experiment_dir = Path(args.model_root) / experiment_name
    args = resolve_args_from_config(args, experiment_dir)

    points_df: pd.DataFrame | None = None
    if args.apply_nearby_filter and not args.summary_only:
        if not args.points_csv:
            raise ValueError("Set --points-csv when --nearby-filter is enabled.")
        points_path = Path(args.points_csv)
        if not points_path.exists():
            raise FileNotFoundError(f"Could not find the representative points file: {points_path}")
        print(f"Loading representative localization points from: {points_path}")
        points_df = pd.read_csv(points_path)
        required_columns = {"image_number", "x", "y"}
        missing_columns = sorted(required_columns - set(points_df.columns))
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"Representative points CSV is missing required column(s): {joined}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Model: {args.model_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Nearby filter: {'enabled' if args.apply_nearby_filter else 'disabled'}")

    summaries: list[dict[str, float]] = []
    output_subdir = get_output_subdir(args.apply_nearby_filter)
    outputs_dir = get_outputs_dir(experiment_dir, args.apply_nearby_filter)
    print(f"Output directory: {outputs_dir}")
    if not args.summary_only:
        for fold in range(1, int(args.k_folds) + 1):
            summary = evaluate_fold(fold, args, experiment_dir, outputs_dir, device, points_df)
            if summary is not None:
                summaries.append(summary)

        if not summaries:
            raise RuntimeError("No fold metrics were produced. Check experiment name and weight files.")

        write_csv(outputs_dir / "test_metrics.csv", summaries)
        (outputs_dir / "test_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        experiment_summary = write_experiment_summary(outputs_dir, summaries)
        (outputs_dir / "visualize_config.json").write_text(
            json.dumps(
                {
                    **asdict(load_visualize_config(args.config)),
                    **vars(args),
                    "experiment_name": experiment_name,
                    "output_subdir": output_subdir,
                    "outputs_dir": str(outputs_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        metric_label = "Test metrics (nearby filtered)" if args.apply_nearby_filter else "Test metrics"
        print(f"\n{metric_label}:")
        for summary in summaries:
            print(
                f"Fold {int(summary['fold'])}: "
                f"Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, "
                f"Precision={summary['precision']:.4f}, Recall={summary['recall']:.4f}"
            )
        print(
            "Average across folds: "
            f"Dice={experiment_summary['mean']['dice']:.4f} +/- {experiment_summary['std']['dice']:.4f}, "
            f"IoU={experiment_summary['mean']['iou']:.4f} +/- {experiment_summary['std']['iou']:.4f}"
        )
        print(f"\nSaved visualizations and metrics under: {outputs_dir}")

    if args.summarize_all_datasets:
        split_name = split_dataset_experiment_name(experiment_name)
        setting_suffix = args.setting_suffix or (split_name[1] if split_name else None)
        if setting_suffix is None:
            raise ValueError("Set --setting-suffix or use an experiment name that starts with 'tu_' or 'drac_'.")
        dataset_rows = summarize_all_datasets(args.model_root, setting_suffix, output_subdir=output_subdir)
        if dataset_rows:
            print("\nAverage performance with fold variation:")
            for row in dataset_rows:
                print(
                    f"{row['dataset']}: "
                    f"Dice={format_summary_metric(row, 'dice')}, "
                    f"IoU={format_summary_metric(row, 'iou')}, "
                    f"Precision={format_summary_metric(row, 'precision')}, "
                    f"Recall={format_summary_metric(row, 'recall')}"
                )
            print(f"Saved dataset summary under: {Path(args.model_root) / 'dataset_summaries'}")


if __name__ == "__main__":
    main()
