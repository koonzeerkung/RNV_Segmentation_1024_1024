"""Generate pixel-wise RNV probability heatmaps from the TU and DRAC U-Net models.

The selected model is an average of its available cross-validation fold
checkpoints. The default output is one dark probability-overlay PNG with a
fixed 0--1 RNV probability scale.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

from utils.dataloader import find_image_file, resolve_dataset_layout
from utils.model_factory import get_model


MODEL_DIRS = {
    "tu": Path("model/BEST_tu_unet_effib3_t0.35"),
    "drac": Path("model/BEST_drac_unet_effib3_t0.35"),
}
MODEL_NAME = "unet"
BACKBONE = "efficientnet-b3"
MULTIPLE_POINTS_CSV = {
    "tu": Path("TU_RNV_Points_Unclustered.csv"),
    "drac": Path("DRAC_RNV_Points_Unclustered.csv"),
}
POINT_IMAGE_TEMPLATES = {"tu": "nv ({sample_id}).jpg", "drac": "{sample_id}.png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an RNV probability-overlay heatmap with a TU or DRAC EfficientNet-B3 U-Net ensemble."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input retinal image path.")
    source.add_argument(
        "--dataset",
        choices=("tu", "drac", "dataset_tu", "dataset_drac"),
        help="Use a project dataset image; requires --sample-id.",
    )
    parser.add_argument("--sample-id", type=int, help="One-based sample ID when using --dataset.")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Dataset root (default: current directory).")
    parser.add_argument(
        "--models",
        choices=("tu", "drac"),
        default=None,
        help="Checkpoint ensemble to run. Defaults to the selected dataset's model.",
    )
    parser.add_argument(
        "--folds",
        default="all",
        help="Comma-separated fold IDs to average, or 'all' (default: all available folds).",
    )
    parser.add_argument("--tu-model-dir", type=Path, default=MODEL_DIRS["tu"])
    parser.add_argument("--drac-model-dir", type=Path, default=MODEL_DIRS["drac"])
    parser.add_argument("--output-dir", type=Path, default=Path("rnv_heatmaps"))
    parser.add_argument(
        "--output-name",
        help="Optional filename for the heatmap PNG, with or without the .png extension.",
    )
    parser.add_argument(
        "--roi-mask",
        type=Path,
        help=(
            "Optional binary mask used as a region of interest. Probabilities outside its white "
            "area are hidden in the saved heatmap overlay."
        ),
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
        help="Square inference size used during training (default: 1024).",
    )
    parser.add_argument(
        "--background-brightness",
        type=float,
        default=0.55,
        help="Raw-image brightness beneath the heatmap, from 0 (black) to 1 (original intensity; default: 0.55).",
    )
    parser.add_argument(
        "--background-gamma",
        type=float,
        default=1.0,
        help="Gamma used to lift dark raw-image detail; values below 1 are brighter (default: 1).",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-probability-npy",
        action="store_true",
        help="Also save the float32 probability array; the default writes only one PNG.",
    )
    parser.add_argument(
        "--compact-points-filter",
        action="store_true",
        help=(
            "Write one compact raw-image / heatmap / multiple-points-filtered-mask panel. "
            "Requires --dataset and uses all unclustered localizer points for that dataset."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.35,
        help="Probability threshold before point filtering (default: 0.35).",
    )
    parser.add_argument(
        "--radius-tolerance",
        type=float,
        default=20.0,
        help="Pixels from a localizer point allowed for a component to remain (default: 20).",
    )
    args = parser.parse_args()
    if args.dataset is not None and args.sample_id is None:
        parser.error("--sample-id is required with --dataset")
    if args.sample_id is not None and args.dataset is None:
        parser.error("--sample-id can only be used with --dataset")
    if args.input_size <= 0:
        parser.error("--input-size must be positive")
    if not 0 <= args.background_brightness <= 1:
        parser.error("--background-brightness must be between 0 and 1")
    if args.background_gamma <= 0:
        parser.error("--background-gamma must be positive")
    if not 0 <= args.mask_threshold <= 1:
        parser.error("--mask-threshold must be between 0 and 1")
    if args.radius_tolerance < 0:
        parser.error("--radius-tolerance must be non-negative")
    if args.compact_points_filter and args.dataset is None:
        parser.error("--compact-points-filter requires --dataset so its multiple-points CSV can be resolved")
    if args.roi_mask is not None and not args.roi_mask.is_file():
        parser.error(f"--roi-mask does not exist: {args.roi_mask}")
    if args.models is None:
        if args.dataset is None:
            parser.error("--models is required when using --image")
        args.models = "tu" if args.dataset in {"tu", "dataset_tu"} else "drac"
    return args


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.image is not None:
        if not args.image.is_file():
            raise FileNotFoundError(f"Input image not found: {args.image}")
        return args.image
    _, _, raw_dir, _ = resolve_dataset_layout(args.dataset, args.base_dir)
    return find_image_file(raw_dir, args.sample_id)


def parse_fold_ids(value: str, weights_dir: Path) -> list[int]:
    available = sorted(int(path.stem) for path in weights_dir.glob("*.pth") if path.stem.isdigit())
    if not available:
        raise FileNotFoundError(f"No numbered .pth checkpoints found in {weights_dir}")
    if value.strip().lower() == "all":
        return available
    try:
        requested = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise ValueError("--folds must be 'all' or comma-separated integer fold IDs") from error
    if not requested:
        raise ValueError("--folds must name at least one fold")
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"Missing checkpoint fold(s) {missing} in {weights_dir}")
    return requested


def load_image_tensor(path: Path, input_size: int) -> tuple[torch.Tensor, tuple[int, int], np.ndarray]:
    with Image.open(path) as source:
        grayscale = source.convert("L")
        original_width, original_height = grayscale.size
        original = np.asarray(grayscale, dtype=np.uint8)
        if grayscale.size != (input_size, input_size):
            grayscale = grayscale.resize((input_size, input_size), Image.Resampling.BILINEAR)
        image = np.asarray(grayscale, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).contiguous()
    return tensor, (original_width, original_height), original


def logits_from_model(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        return output["out"]
    return output


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load plain state dictionaries from both old and new PyTorch releases."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not support weights_only.
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dictionary: {path}")
    return state


@torch.no_grad()
def predict_ensemble(
    image: torch.Tensor,
    weights_dir: Path,
    folds: Iterable[int],
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Return the mean sigmoid probability across requested fold checkpoints."""
    model = get_model(
        device,
        model_name=MODEL_NAME,
        backbone=BACKBONE,
        in_channels=1,
        out_channels=1,
        encoder_weights=None,
    )
    image = image.to(device, non_blocking=True)
    probability_sum: torch.Tensor | None = None
    fold_list = list(folds)
    model.eval()
    for fold in fold_list:
        checkpoint = weights_dir / f"{fold}.pth"
        model.load_state_dict(load_state_dict(checkpoint, device))
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            probability = torch.sigmoid(logits_from_model(model, image))
        probability = probability.float()
        probability_sum = probability if probability_sum is None else probability_sum + probability
    assert probability_sum is not None
    return (probability_sum / len(fold_list))[0, 0].cpu().numpy().astype(np.float32)


def restore_size(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if probability.shape == (height, width):
        return probability
    tensor = torch.from_numpy(probability).unsqueeze(0).unsqueeze(0)
    restored = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return restored[0, 0].numpy().astype(np.float32)


def load_roi_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a binary ROI, resizing it without interpolation to the image dimensions."""
    height, width = shape
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L"), dtype=np.uint8) > 127
    if mask.shape != (height, width):
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized, dtype=np.uint8) > 127
    return mask


def save_probability_overlay(
    probability: np.ndarray,
    output_path: Path,
    image: np.ndarray,
    label: str,
    roi_mask: np.ndarray | None = None,
    background_brightness: float = 0.55,
    background_gamma: float = 1.0,
) -> None:
    """Save a probability overlay with a fixed 0--1 heatmap scale."""
    fig, axis = plt.subplots(figsize=(8, 8), facecolor="white")
    fig.patch.set_facecolor("white")
    axis.set_facecolor("black")
    # Keep the fundus clearly visible while letting high-probability regions dominate.
    raw_image = image.astype(np.float32) / 255.0
    background_image = np.power(raw_image, background_gamma) * background_brightness
    axis.imshow(background_image, cmap="gray", vmin=0, vmax=1)
    # Low probabilities remain transparent; high probabilities become vivid.
    alpha = 0.10 + 0.80 * np.clip(probability, 0, 1)
    if roi_mask is not None:
        # Match the standard overlay's subtle low-probability blue background
        # without exposing probabilities outside the supplied ROI.
        axis.imshow(np.full_like(probability, 0.10), cmap="turbo", vmin=0, vmax=1, alpha=0.10)
        alpha = np.where(roi_mask, alpha, 0.0)
    shown = axis.imshow(probability, cmap="turbo", vmin=0, vmax=1, alpha=alpha)
    axis.set_title(f"RNV probability overlay: {label}", color="black")
    axis.axis("off")
    colorbar = fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("P(RNV)", color="black")
    colorbar.ax.tick_params(colors="black")
    colorbar.outline.set_edgecolor("black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def resolve_heatmap_output_path(
    output_dir: Path,
    output_name: str | None,
    default_name: str,
) -> Path:
    """Return a PNG output path, disallowing path traversal in a custom name."""
    if not output_name:
        return output_dir / default_name
    filename = Path(output_name).name
    if filename in {"", ".", ".."}:
        raise ValueError("--output-name must include a filename")
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    return output_dir / filename


def filter_mask_by_multiple_points(
    model_mask: np.ndarray,
    image_name: str,
    points_path: Path,
    radius_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the project's nearby-component filter using every localizer point."""
    if not points_path.is_file():
        raise FileNotFoundError(f"Multiple-points CSV not found: {points_path}")
    points_df = pd.read_csv(points_path)
    if "image_number" not in points_df.columns and "image_name" in points_df.columns:
        points_df = points_df.rename(columns={"image_name": "image_number"})
    required_columns = {"image_number", "x", "y"}
    missing = required_columns - set(points_df.columns)
    if missing:
        raise ValueError(f"{points_path} is missing columns: {', '.join(sorted(missing))}")
    image_points = points_df[points_df["image_number"] == image_name]
    if image_points.empty:
        raise ValueError(f"No multiple localizer points for {image_name} in {points_path}")

    ref_points = image_points[["y", "x"]].to_numpy(dtype=np.float64)
    labeled_mask, _ = label(model_mask)
    rows = np.clip(np.rint(ref_points[:, 0]).astype(np.int64), 0, model_mask.shape[0] - 1)
    cols = np.clip(np.rint(ref_points[:, 1]).astype(np.int64), 0, model_mask.shape[1] - 1)
    points_mask = np.zeros_like(model_mask, dtype=bool)
    points_mask[rows, cols] = True
    within_radius = distance_transform_edt(~points_mask) <= radius_tolerance
    valid_labels = np.unique(labeled_mask[within_radius])
    filtered_mask = np.isin(labeled_mask, valid_labels[valid_labels > 0]).astype(np.uint8)
    return filtered_mask, ref_points


def make_model_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold the probability map and fill interior holes, matching evaluation."""
    return binary_fill_holes(probability > threshold).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a pixel-aligned binary mask as a standard black-and-white PNG."""
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)


def save_compact_points_panel(
    image: np.ndarray,
    probability: np.ndarray,
    filtered_mask: np.ndarray,
    reference_points: np.ndarray,
    output_path: Path,
    model_name: str,
    sample_id: int,
) -> None:
    """Save a compact raw / probability / multiple-points-filtered-mask panel."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.patch.set_facecolor("white")

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Raw image")
    axes[0].axis("off")

    heatmap = axes[1].imshow(probability, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("RNV probability heatmap")
    axes[1].axis("off")
    colorbar = fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("P(RNV)")

    axes[2].imshow(filtered_mask, cmap="gray", vmin=0, vmax=1)
    """axes[2].scatter(
        reference_points[:, 1],
        reference_points[:, 0],
        c="red",
        s=20,
        marker="o",
        edgecolors="white",
        linewidths=0.4,
    )"""
    #axes[2].set_title(f"After filtering ({len(reference_points)} points)")
    axes[2].set_title(f"After filtering")
    axes[2].axis("off")

    #fig.suptitle(f"TU sample {sample_id}: {model_name.upper()} model", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    image, original_size, original_image = load_image_tensor(input_path, args.input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {"tu": args.tu_model_dir, "drac": args.drac_model_dir}
    model_name = args.models
    weights_dir = model_dirs[model_name] / "weights"
    folds = parse_fold_ids(args.folds, weights_dir)
    print(f"Running {model_name.upper()} model with folds {folds}...")
    probability = restore_size(
        predict_ensemble(image, weights_dir, folds, device, args.amp), original_size
    )
    model_mask = make_model_mask(probability, args.mask_threshold)
    model_mask_path = output_dir / f"{input_path.stem}_{model_name}_rnv_prediction_mask.png"
    save_binary_mask(model_mask, model_mask_path)
    print(f"Saved model prediction mask: {model_mask_path}")
    if args.compact_points_filter:
        dataset_key = "tu" if args.dataset in {"tu", "dataset_tu"} else "drac"
        image_name = POINT_IMAGE_TEMPLATES[dataset_key].format(sample_id=args.sample_id)
        filtered_mask, reference_points = filter_mask_by_multiple_points(
            model_mask=model_mask,
            image_name=image_name,
            points_path=MULTIPLE_POINTS_CSV[dataset_key],
            radius_tolerance=args.radius_tolerance,
        )
        output_path = output_dir / f"{input_path.stem}_{model_name}_rnv_compact_multiple_points.png"
        filtered_mask_path = output_dir / f"{input_path.stem}_{model_name}_rnv_multiple_points_filtered_mask.png"
        save_binary_mask(filtered_mask, filtered_mask_path)
        save_compact_points_panel(
            image=original_image,
            probability=probability,
            filtered_mask=filtered_mask,
            reference_points=reference_points,
            output_path=output_path,
            model_name=model_name,
            sample_id=args.sample_id,
        )
        print(f"Saved compact multiple-points-filtered panel: {output_path}")
        print(f"Saved multiple-points-filtered model mask: {filtered_mask_path}")
        return
    roi_mask: np.ndarray | None = None
    if args.roi_mask is not None:
        roi_mask = load_roi_mask(args.roi_mask, probability.shape)
        default_name = f"{input_path.stem}_{model_name}_rnv_masked_heatmap.png"
        label = f"{model_name.upper()} model (custom mask)"
    else:
        default_name = f"{input_path.stem}_{model_name}_rnv_probability_overlay.png"
        label = f"{model_name.upper()} model"
    output_path = resolve_heatmap_output_path(output_dir, args.output_name, default_name)
    save_probability_overlay(
        probability,
        output_path,
        original_image,
        label,
        roi_mask=roi_mask,
        background_brightness=args.background_brightness,
        background_gamma=args.background_gamma,
    )
    if args.save_probability_npy:
        probability_path = output_path.with_suffix(".npy")
        np.save(probability_path, probability)
        print(f"Saved probability array: {probability_path}")
    print(f"Saved RNV probability overlay: {output_path}")


if __name__ == "__main__":
    main()
