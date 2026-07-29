"""Re-score existing trained weights against raw GT vs hole-filled GT.

Inference only — no training. Answers "how much do the metrics move if the ground truth
gets the same `binary_fill_holes` pass the predictions already get?" for models that were
trained on raw GT.

Usage:
    python rescore_gt_fill_holes.py                       # all model/BEST_*eff* experiments
    python rescore_gt_fill_holes.py --experiments BEST_tu_unet_effib3_t0.35
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from torch.utils.data import DataLoader

from utils.dataloader import KFoldSliceDataset, build_samples
from utils.metrics import compute_binary_mask_metrics, get_postprocessed_mask
from utils.model_factory import get_model

METRIC_KEYS = ("dice", "iou", "precision", "recall", "specificity", "accuracy")


def get_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    return output["out"] if isinstance(output, dict) else output


def fill_target(target: torch.Tensor) -> torch.Tensor:
    """Apply the prediction-side hole filling to a GT mask tensor."""

    array = target[0, 0].numpy() > 0.5
    return torch.from_numpy(binary_fill_holes(array).astype(np.float32)).unsqueeze(0).unsqueeze(0)


def evaluate_experiment(
    experiment_dir: Path,
    device: torch.device,
    use_amp: bool,
    k_folds: int,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        print(f"[skip] {experiment_dir.name}: no config.json")
        return None

    config = json.loads(config_path.read_text())
    dataset = config["dataset"]
    model_name = config["model_name"]
    backbone = config["backbone"]
    threshold = float(config.get("threshold", 0.5))

    print(f"\n=== {experiment_dir.name} ({dataset}/{model_name}/{backbone}, threshold={threshold})")

    model = get_model(device, model_name=model_name, backbone=backbone)
    rows: list[dict[str, object]] = []

    for fold in range(1, k_folds + 1):
        weights_path = experiment_dir / "weights" / f"{fold}.pth"
        if not weights_path.exists():
            print(f"  [skip] fold {fold}: missing {weights_path.name}")
            continue

        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        # gt_fill_holes=False so the loader hands back raw GT; filling happens per-item below.
        samples = build_samples(dataset=dataset, fold=fold, base_dir=config.get("base_dir", "."))["test"]
        loader = DataLoader(
            KFoldSliceDataset(samples, augment="no", include_metadata=True, gt_fill_holes=False),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                    logits = get_logits(model, images)

                pred = get_postprocessed_mask(logits.float(), threshold=threshold).cpu()
                raw_gt = batch["mask"]
                filled_gt = fill_target(raw_gt)

                raw_metrics = compute_binary_mask_metrics(pred, raw_gt)
                filled_metrics = compute_binary_mask_metrics(pred, filled_gt)

                row: dict[str, object] = {
                    "experiment": experiment_dir.name,
                    "dataset": dataset,
                    "fold": fold,
                    "sample_id": int(batch["sample_id"][0]),
                    "gt_px_raw": int(raw_gt.sum().item()),
                    "gt_px_filled": int(filled_gt.sum().item()),
                    "pred_px": int(pred.sum().item()),
                }
                for key in METRIC_KEYS:
                    row[f"{key}_raw_gt"] = float(raw_metrics[key])
                    row[f"{key}_filled_gt"] = float(filled_metrics[key])
                    row[f"{key}_delta"] = row[f"{key}_filled_gt"] - row[f"{key}_raw_gt"]
                rows.append(row)

        print(f"  fold {fold}: {len(loader)} images scored")

    if not rows:
        return None

    summary: dict[str, object] = {
        "experiment": experiment_dir.name,
        "dataset": dataset,
        "model_name": model_name,
        "backbone": backbone,
        "threshold": threshold,
        "images": len(rows),
    }
    for key in METRIC_KEYS:
        summary[f"{key}_raw_gt"] = float(np.mean([row[f"{key}_raw_gt"] for row in rows]))
        summary[f"{key}_filled_gt"] = float(np.mean([row[f"{key}_filled_gt"] for row in rows]))
        summary[f"{key}_delta"] = summary[f"{key}_filled_gt"] - summary[f"{key}_raw_gt"]

    print(
        f"  --> dice {summary['dice_raw_gt']:.4f} -> {summary['dice_filled_gt']:.4f} "
        f"({summary['dice_delta']:+.4f}), precision {summary['precision_raw_gt']:.4f} -> "
        f"{summary['precision_filled_gt']:.4f} ({summary['precision_delta']:+.4f})"
    )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="model")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="experiment dir names; default is every model/BEST_*eff* dir",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="gt_fillhole_check/rescore")
    args = parser.parse_args()

    model_root = Path(args.model_root)
    if args.experiments:
        experiment_dirs = [model_root / name for name in args.experiments]
    else:
        experiment_dirs = sorted(
            path for path in model_root.glob("BEST_*eff*") if path.is_dir()
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiments: {[path.name for path in experiment_dirs]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for experiment_dir in experiment_dirs:
        result = evaluate_experiment(experiment_dir, device, args.amp, args.k_folds)
        if result is None:
            continue
        rows, summary = result
        all_rows.extend(rows)
        summaries.append(summary)

    if not summaries:
        print("nothing scored")
        return

    with (out_dir / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    print("\n=== raw GT -> filled GT (mean over test images, no nearby filter) ===")
    header = f"{'experiment':<42} {'dice':>22} {'precision':>22} {'recall':>22}"
    print(header)
    for summary in summaries:
        cells = "".join(
            f"  {summary[f'{key}_raw_gt']:.4f}->{summary[f'{key}_filled_gt']:.4f}({summary[f'{key}_delta']:+.4f})"
            for key in ("dice", "precision", "recall")
        )
        print(f"{summary['experiment']:<42}{cells}")
    print(f"\nwrote {out_dir}/summary.csv and per_image.csv")


if __name__ == "__main__":
    main()
