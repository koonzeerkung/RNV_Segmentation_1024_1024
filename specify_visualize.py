from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from PIL import Image

from utils.dataloader import SOURCE_SIZE, find_image_file, resolve_dataset_layout
from utils.metrics import compute_binary_mask_metrics, get_postprocessed_mask
from utils.model_factory import get_model, normalize_backbone_name, normalize_encoder_weights, normalize_model_name


# =========================
# Parameter section
# =========================
# Edit these defaults when you want to run without positional CLI arguments.
DEFAULT_WEIGHT_PATH: str | None = "model/tu_0.4thresold_100epoch_efficientnet/weights/1.pth"
DEFAULT_DATASET: str | None = "tu"
DEFAULT_SAMPLE_ID: int | None = 1
DEFAULT_BASE_DIR: str | None = "."
DEFAULT_THRESHOLD: float | None = 0.4
DEFAULT_MODEL_NAME: str | None = None
DEFAULT_BACKBONE: str | None = None
DEFAULT_ENCODER_WEIGHTS: str | None = None
DEFAULT_USE_AMP = True
OUTPUT_SUBDIR = "extra_output"
SAVE_PREDICTION_NPY = True
FIGURE_DPI = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one dataset image using one trained weight file."
    )
    parser.add_argument(
        "weight_path",
        nargs="?",
        default=DEFAULT_WEIGHT_PATH,
        help="Path to a trained fold weight file, usually model/<experiment>/weights/<fold>.pth.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        choices=["tu", "drac", "dataset_tu", "dataset_drac"],
    )
    parser.add_argument(
        "sample_id",
        nargs="?",
        type=int,
        default=DEFAULT_SAMPLE_ID,
        help="Dataset image/sample id, for example 1.",
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Project/data root containing dataset_tu and dataset_drac.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model architecture: unet, unetplusplus, deeplabv3plus, or deeplabv3.",
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        help="Encoder backbone: efficientnet-b3, inceptionv4, or densenet169.",
    )
    parser.add_argument(
        "--encoder-weights",
        default=DEFAULT_ENCODER_WEIGHTS,
        help="Encoder weights passed to segmentation-models-pytorch. Use 'none' to avoid downloads.",
    )
    parser.add_argument("--amp", "--use-amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_AMP)
    args = parser.parse_args()
    missing = [
        name
        for name in ("weight_path", "dataset", "sample_id")
        if getattr(args, name) is None
    ]
    if missing:
        joined = ", ".join(missing)
        parser.error(f"Missing required value(s): {joined}. Set them in the parameter section or pass them as CLI arguments.")
    if args.model_name:
        args.model_name = normalize_model_name(args.model_name)
    if args.backbone:
        args.backbone = normalize_backbone_name(args.backbone)
    args.encoder_weights = normalize_encoder_weights(args.encoder_weights)
    return args


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


def infer_experiment_dir(weight_path: Path) -> Path:
    if weight_path.parent.name == "weights":
        return weight_path.parent.parent

    parts = weight_path.parts
    if "model" in parts:
        model_index = parts.index("model")
        if model_index + 1 < len(parts):
            return Path(*parts[: model_index + 2])

    return Path("model") / weight_path.stem


def load_experiment_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).contiguous()


def mask_to_tensor(mask: Image.Image, threshold: float = 0.5) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    array = (array >= threshold).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).contiguous()


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
    target: torch.Tensor | None,
    pred: torch.Tensor,
    save_path: Path,
    title: str,
) -> None:
    image_np = image[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    if target is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title)
        axes[0].imshow(image_np, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Prediction")
        axes[1].axis("off")
    else:
        target_np = target[0, 0].detach().cpu().numpy()
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
        axes[1].imshow(target_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
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

        text_threshold = cm_data.max() / 2.0 if cm_data.max() else 0
        for row, col in np.ndindex(cm_data.shape):
            color = "white" if cm_data[row, col] > text_threshold else "black"
            axes[4].text(
                col,
                row,
                f"{cm_data[row, col]:,}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    weight_path = Path(args.weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing weight file: {weight_path}")

    experiment_dir = infer_experiment_dir(weight_path)
    config = load_experiment_config(experiment_dir)
    base_dir = args.base_dir if args.base_dir is not None else str(config.get("base_dir", "."))
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    model_name = args.model_name or normalize_model_name(str(config.get("model_name", "unet")))
    backbone = args.backbone or normalize_backbone_name(str(config.get("backbone", "efficientnet-b3")))
    encoder_weights = (
        args.encoder_weights
        if args.encoder_weights is not None
        else normalize_encoder_weights(config.get("encoder_weights", "imagenet"))
    )

    dataset_name, _, raw_dir, gt_dir = resolve_dataset_layout(args.dataset, base_dir)
    raw_path = find_image_file(raw_dir, args.sample_id)
    gt_path: Path | None = None
    if gt_dir.is_dir():
        try:
            gt_path = find_image_file(gt_dir, args.sample_id)
        except FileNotFoundError:
            gt_path = None

    with Image.open(raw_path) as raw_image:
        raw_image = raw_image.convert("L")
        if raw_image.size != (SOURCE_SIZE, SOURCE_SIZE):
            raise ValueError(f"Expected {SOURCE_SIZE}x{SOURCE_SIZE} image, got {raw_image.size}: {raw_path}")

        gt_image: Image.Image | None = None
        if gt_path is not None:
            gt_image = Image.open(gt_path).convert("L")
            if gt_image.size != (SOURCE_SIZE, SOURCE_SIZE):
                raise ValueError(f"Expected {SOURCE_SIZE}x{SOURCE_SIZE} mask, got {gt_image.size}: {gt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(
            device,
            model_name=model_name,
            backbone=backbone,
            encoder_weights=encoder_weights,
            in_channels=1,
            out_channels=1,
        )
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        use_amp = bool(args.use_amp and device.type == "cuda")
        image_tensor = image_to_tensor(raw_image)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = get_logits(model, image_tensor.to(device, non_blocking=True))
        if logits.shape[-2:] != image_tensor.shape[-2:]:
            raise ValueError(
                f"Expected model output spatial size {tuple(image_tensor.shape[-2:])}, got {tuple(logits.shape[-2:])}"
            )

        full_input = image_tensor.cpu()
        full_mask = mask_to_tensor(gt_image) if gt_image is not None else None
        full_logits = logits.float().detach().cpu()

        if gt_image is not None:
            gt_image.close()

    pred = get_postprocessed_mask(full_logits, threshold=threshold)
    metrics: dict[str, Any] | None = None
    if full_mask is not None:
        metrics = compute_binary_mask_metrics(pred, full_mask)
        metrics.update({
            "sample_id": args.sample_id,
            "dataset": dataset_name,
            "weight_path": str(weight_path),
            "threshold": threshold,
        })

    output_dir = experiment_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_stem = f"{dataset_name}_sample_{args.sample_id}"
    visualization_path = output_dir / f"{sample_stem}.png"
    prediction_path = output_dir / f"{sample_stem}_prediction.npy"

    save_visualization(
        image=full_input,
        target=full_mask,
        pred=pred,
        save_path=visualization_path,
        title=f"{dataset_name} sample {args.sample_id}",
    )
    if SAVE_PREDICTION_NPY:
        np.save(prediction_path, pred[0, 0].detach().cpu().numpy().astype(np.uint8))
    if metrics is not None:
        write_csv(output_dir / f"{sample_stem}_metrics.csv", [metrics])

    print(f"Experiment directory: {experiment_dir}")
    print(f"Weight: {weight_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Sample ID: {args.sample_id}")
    print(f"Model: {model_name}")
    print(f"Backbone: {backbone}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold:g}")
    if metrics is not None:
        print(
            "Metrics: "
            f"Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
            f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )
    print(f"Saved visualization: {visualization_path}")
    if SAVE_PREDICTION_NPY:
        print(f"Saved prediction mask: {prediction_path}")


if __name__ == "__main__":
    main()
