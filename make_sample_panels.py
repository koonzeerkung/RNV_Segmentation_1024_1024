"""
Render the six requested per-sample panels separately (not as one combined figure), for
specific (dataset, sample_id) pairs across a set of RETRAIN_* models.

visualize_nearby.py only ever saves one combined matplotlib figure per sample
(save_visualization()); there is no raw array on disk to reuse, so this script re-runs
inference for exactly the requested samples and renders each panel to its own PNG:

  01_raw_prediction.png              thresholded, hole-filled prediction, no filter
  02_raw_prediction_with_points.png  same, with the candidate points overlaid
  03_ground_truth_with_points.png    ground truth mask, with the candidate points overlaid
  04_nearby_prediction.png           prediction after the candidate-point filter
  05_error_map.png                   TP/FP/FN of the nearby prediction vs ground truth
  06_nearby_overlay_on_input.png     nearby prediction drawn semi-transparently over the
                                      raw OCTA image, so the underlying vessels stay visible

Standalone module: every setting lives in the CONFIG section below, not on the command line.
Edit the constants and run:

    python make_sample_panels.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from utils.dataloader import create_dataloaders, get_kfold_chunks
from utils.dataset_points import DATASET_POINTS_IMAGE_NAME_TEMPLATE, resolve_points_csv
from utils.metrics import get_postprocessed_mask
from utils.model_factory import get_model
from visualize_nearby import filter_by_representative_points, get_logits

# ============================================================================
# CONFIG -- edit these, then run: python make_sample_panels.py
# ============================================================================

# (dataset, sample_id) pairs to render.
SAMPLES: list[tuple[str, int]] = [
    ("tu", 16),
    ("tu", 23),
    ("tu", 27),
    ("drac", 3),
    ("drac", 16),
    ("drac", 37),
]

# (short label used in filenames, TU experiment dir, DRAC experiment dir).
MODELS: list[tuple[str, str, str]] = [
    ("unet", "RETRAIN_tu_unet_effib3_t0.35", "RETRAIN_drac_unet_effib3_t0.35"),
    ("unetplusplus", "RETRAIN_tu_unetplusplus_effib3_t0.35", "RETRAIN_drac_unetplusplus_effib3_t0.35"),
    ("deeplabv3plus", "RETRAIN_tu_deeplabv3plus_efficientnetb3", "RETRAIN_drac_deeplabv3plus_efficientnetb3"),
]

# "multiple" = every localizer point (pipeline default); "single" = one point per lesion.
POINTS_MODE = "multiple"

# Threshold and backbone are read per-experiment from its own config.json; this is only
# the candidate-point radius, which is not stored in config.json (it's a visualize.py CLI
# default) and is 20px in every config in this project.
RADIUS_TOLERANCE = 20.0

MODEL_ROOT = Path("model")
OUTPUT_ROOT = Path("sample_panels")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OVERLAY_COLOR = (1.0, 0.0, 0.0)  # red
OVERLAY_ALPHA = 0.45

# ============================================================================


def fold_for_sample(dataset: str, sample_id: int) -> int:
    chunks = get_kfold_chunks(dataset)
    for fold_index, chunk in enumerate(chunks, start=1):
        if sample_id in chunk:
            return fold_index
    raise ValueError(f"sample {sample_id} not in any fold of dataset {dataset}")


def load_points(dataset: str, points_mode: str) -> pd.DataFrame:
    points_path = Path(resolve_points_csv(dataset, points_mode))
    if not points_path.exists():
        raise FileNotFoundError(f"points CSV not found: {points_path}")
    points_df = pd.read_csv(points_path)
    if "image_number" not in points_df.columns and "image_name" in points_df.columns:
        points_df = points_df.rename(columns={"image_name": "image_number"})
    return points_df


def save_gray(mask_np: np.ndarray, path: Path, points: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    if points is not None and len(points) > 0:
        ax.scatter(points[:, 1], points[:, 0], c="red", s=60, marker="o",
                    edgecolors="white", linewidths=1.0)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


def save_error_map(pred_np: np.ndarray, target_np: np.ndarray, path: Path) -> None:
    error_map = np.zeros_like(target_np)
    error_map[(pred_np == 1) & (target_np == 1)] = 1  # TP
    error_map[(pred_np == 1) & (target_np == 0)] = 2  # FP
    error_map[(pred_np == 0) & (target_np == 1)] = 3  # FN
    error_cmap = ListedColormap(["black", "lime", "red", "blue"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(error_map, cmap=error_cmap, vmin=0, vmax=3)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


def save_overlay(image_np: np.ndarray, pred_np: np.ndarray, path: Path) -> None:
    """Nearby-filtered prediction drawn over the raw image with alpha, background transparent."""

    overlay_rgba = np.zeros((*pred_np.shape, 4), dtype=np.float32)
    overlay_rgba[..., 0] = OVERLAY_COLOR[0]
    overlay_rgba[..., 1] = OVERLAY_COLOR[1]
    overlay_rgba[..., 2] = OVERLAY_COLOR[2]
    overlay_rgba[..., 3] = pred_np.astype(np.float32) * OVERLAY_ALPHA

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, cmap="gray")
    ax.imshow(overlay_rgba)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


def render_for_model(
    model_key: str,
    dataset: str,
    experiment_dir: Path,
    samples: list[int],
    points_df: pd.DataFrame,
) -> None:
    config = json.loads((experiment_dir / "config.json").read_text())
    threshold = float(config["threshold"])
    backbone = config["backbone"]
    model_name = config["model_name"]
    base_dir = config.get("base_dir", ".")
    use_amp = bool(config.get("use_amp", True))
    name_template = DATASET_POINTS_IMAGE_NAME_TEMPLATE[dataset]

    samples_by_fold: dict[int, list[int]] = {}
    for sample_id in samples:
        samples_by_fold.setdefault(fold_for_sample(dataset, sample_id), []).append(sample_id)

    for fold, fold_samples in samples_by_fold.items():
        weights_path = experiment_dir / "weights" / f"{fold}.pth"
        if not weights_path.exists():
            print(f"  [skip] {experiment_dir.name} fold {fold}: weights missing")
            continue

        loaders = create_dataloaders(
            dataset=dataset,
            augment=False,
            aug_multiplier=1,
            fold=fold,
            batch_size=1,
            base_dir=base_dir,
            num_workers=0,
            include_metadata=True,
            pin_memory=DEVICE.type == "cuda",
        )
        test_loader = loaders["test"]

        model = get_model(DEVICE, model_name=model_name, backbone=backbone, in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()

        remaining = set(fold_samples)
        with torch.no_grad():
            for batch in test_loader:
                sample_id = int(batch["sample_id"][0])
                if sample_id not in remaining:
                    continue

                images = batch["image"].to(DEVICE, non_blocking=True)
                masks = batch["mask"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp and DEVICE.type == "cuda"):
                    logits = get_logits(model, images)

                raw_pred = get_postprocessed_mask(logits.float(), threshold=threshold)
                image_number = name_template.format(sample_id=sample_id)
                final_pred, ref_points = filter_by_representative_points(
                    pred_mask=raw_pred,
                    image_name=image_number,
                    points_df=points_df,
                    radius_tolerance=RADIUS_TOLERANCE,
                )

                image_np = images[0, 0].detach().cpu().numpy()
                target_np = masks[0, 0].detach().cpu().numpy()
                raw_pred_np = raw_pred[0, 0].detach().cpu().numpy()
                final_pred_np = final_pred[0, 0].detach().cpu().numpy()

                sample_dir = OUTPUT_ROOT / f"{dataset}_sample_{sample_id}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                prefix = model_key

                save_gray(raw_pred_np, sample_dir / f"{prefix}_01_raw_prediction.png")
                save_gray(raw_pred_np, sample_dir / f"{prefix}_02_raw_prediction_with_points.png", ref_points)
                save_gray(target_np, sample_dir / f"{prefix}_03_ground_truth_with_points.png", ref_points)
                save_gray(final_pred_np, sample_dir / f"{prefix}_04_nearby_prediction.png")
                save_error_map(final_pred_np, target_np, sample_dir / f"{prefix}_05_error_map.png")
                save_overlay(image_np, final_pred_np, sample_dir / f"{prefix}_06_nearby_overlay_on_input.png")

                print(f"  {dataset} sample {sample_id} fold {fold} [{model_key}] "
                      f"-> {sample_dir} ({'0' if ref_points is None else len(ref_points)} points)")

                remaining.discard(sample_id)
                if not remaining:
                    break

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        if remaining:
            print(f"  [warn] {experiment_dir.name} fold {fold}: samples not found in test split: {sorted(remaining)}")


def main() -> None:
    samples_by_dataset: dict[str, list[int]] = {}
    for dataset, sample_id in SAMPLES:
        samples_by_dataset.setdefault(dataset, []).append(sample_id)

    points_cache: dict[str, pd.DataFrame] = {}

    for model_key, tu_exp, drac_exp in MODELS:
        for dataset, exp_name in (("tu", tu_exp), ("drac", drac_exp)):
            samples = samples_by_dataset.get(dataset)
            if not samples:
                continue
            experiment_dir = MODEL_ROOT / exp_name
            if not experiment_dir.exists():
                print(f"[skip] {experiment_dir} does not exist")
                continue

            if dataset not in points_cache:
                points_cache[dataset] = load_points(dataset, POINTS_MODE)

            print(f"{exp_name}  (points_mode={POINTS_MODE})")
            render_for_model(
                model_key=model_key,
                dataset=dataset,
                experiment_dir=experiment_dir,
                samples=samples,
                points_df=points_cache[dataset],
            )

    print(f"\nwrote panels under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
