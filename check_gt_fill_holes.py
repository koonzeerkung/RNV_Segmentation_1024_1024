"""Apply the prediction-side hole filling to ground-truth masks and report the delta.

Mirrors `utils.metrics.get_postprocessed_mask` (sigmoid/threshold replaced by the
dataloader's own mask binarization) so the before/after comparison uses exactly the
postprocessing that predictions already receive.

Usage:
    python check_gt_fill_holes.py                    # TU dataset, default output dir
    python check_gt_fill_holes.py --dataset drac
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label

from utils.dataloader import find_image_file, get_kfold_chunks, resolve_dataset_layout

MASK_THRESHOLD = 0.5


def binarize(mask_path: Path) -> np.ndarray:
    """Binarize a GT file the same way KFoldSliceDataset._mask_to_tensor does."""

    with Image.open(mask_path) as mask:
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return (array.astype(np.float32) / 255.0) >= MASK_THRESHOLD


def panel(
    raw: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    added: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))

    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("raw image")

    axes[1].imshow(before, cmap="gray")
    axes[1].set_title(f"GT before fill ({int(before.sum())} px)")

    axes[2].imshow(after, cmap="gray")
    axes[2].set_title(f"GT after fill ({int(after.sum())} px)")

    overlay = np.zeros((*before.shape, 3), dtype=np.float32)
    overlay[before] = (1.0, 1.0, 1.0)
    overlay[added] = (1.0, 0.0, 0.0)
    axes[3].imshow(overlay)
    axes[3].set_title(f"added by fill (red, {int(added.sum())} px)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tu", help="tu or drac")
    parser.add_argument("--output-dir", default="", help="default: gt_fillhole_check/<dataset>")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--no-panels", action="store_true", help="stats only, skip PNGs")
    args = parser.parse_args()

    dataset_name, _, raw_dir, gt_dir = resolve_dataset_layout(args.dataset, args.base_dir)
    short = dataset_name.replace("dataset_", "")
    out_dir = Path(args.output_dir) if args.output_dir else Path("gt_fillhole_check") / short
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [sample_id for chunk in get_kfold_chunks(dataset_name) for sample_id in chunk]

    rows: list[dict[str, object]] = []
    for sample_id in sorted(sample_ids):
        mask_path = find_image_file(gt_dir, sample_id)
        image_path = find_image_file(raw_dir, sample_id)

        before = binarize(mask_path)
        after = binary_fill_holes(before)
        added = after & ~before

        _, hole_count = label(added)
        hole_sizes = np.bincount(label(added)[0].ravel())[1:] if hole_count else np.array([])

        row = {
            "sample_id": sample_id,
            "gt_file": mask_path.name,
            "px_before": int(before.sum()),
            "px_after": int(after.sum()),
            "px_added": int(added.sum()),
            "pct_added": round(100.0 * added.sum() / max(before.sum(), 1), 4),
            "holes_filled": int(hole_count),
            "largest_hole_px": int(hole_sizes.max()) if hole_sizes.size else 0,
        }
        rows.append(row)

        if not args.no_panels:
            with Image.open(image_path) as raw_image:
                raw = np.asarray(raw_image.convert("L"), dtype=np.uint8)
            title = (
                f"{short} sample {sample_id} — {row['px_before']} -> {row['px_after']} px "
                f"(+{row['px_added']}, +{row['pct_added']}%), {row['holes_filled']} holes filled"
            )
            panel(raw, before, after, added, title, out_dir / f"gt_fill_{sample_id:02d}.png")

        print(
            f"[{sample_id:>3}] {mask_path.name:<16} before={row['px_before']:>8} "
            f"after={row['px_after']:>8} added={row['px_added']:>7} "
            f"(+{row['pct_added']:>7}%) holes={row['holes_filled']:>4} "
            f"largest={row['largest_hole_px']:>6}"
        )

    fieldnames = list(rows[0].keys())
    csv_path = out_dir / "gt_fill_holes_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_before = sum(int(row["px_before"]) for row in rows)
    total_added = sum(int(row["px_added"]) for row in rows)
    changed = [row for row in rows if int(row["px_added"]) > 0]
    summary = {
        "dataset": short,
        "images": len(rows),
        "images_changed": len(changed),
        "total_px_before": total_before,
        "total_px_added": total_added,
        "total_pct_added": round(100.0 * total_added / max(total_before, 1), 4),
        "total_holes_filled": sum(int(row["holes_filled"]) for row in rows),
        "max_pct_added_image": max(rows, key=lambda row: row["pct_added"])["sample_id"] if rows else None,
    }
    (out_dir / "gt_fill_holes_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nwrote {csv_path} and panels to {out_dir}")


if __name__ == "__main__":
    main()
