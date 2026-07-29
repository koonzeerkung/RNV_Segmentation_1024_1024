"""
Select the binarization threshold on the VALIDATION folds, never the test folds.

Motivation: the thresholds shipped in configs/best_*.toml (0.35 for U-Net / U-Net++,
0.5 for DeepLabv3+) were chosen by inspecting predictions. `visualize_nearby.py` renders
the *test* split, so any threshold picked that way is selection on the test set. This
script re-derives the threshold from the validation split of each fold using the existing
trained weights, so the reported number has a defensible provenance.

No retraining. Logits are computed once per validation image and reused for every
threshold in the grid, so the sweep costs one forward pass per image regardless of grid
size.

Aggregation matches the paper's evaluation protocol (ref/methodology.tex, subsec:eval):
per-image metric -> mean within a fold -> mean/std across folds.

Usage:
    python select_threshold_val.py --experiment BEST_tu_unet_effib3_t0.35
    python select_threshold_val.py --experiment BEST_drac_unet_effib3_t0.35 \
        --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60
    python select_threshold_val.py --all-best          # every BEST_* dir with a config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.dataloader import create_dataloaders
from utils.metrics import compute_binary_mask_metrics, get_postprocessed_mask
from utils.model_factory import get_model

DEFAULT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
METRICS = ["dice", "iou", "precision", "recall"]


def get_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract logits from the wrapped model (mirrors main.py / visualize_nearby.py)."""

    output = model(images)
    if isinstance(output, dict):
        return output["out"]
    return output


def parse_thresholds(raw: str) -> list[float]:
    values = [float(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("no thresholds parsed")
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold {value} outside (0, 1)")
    return values


def sweep_fold(
    *,
    weights_path: Path,
    dataset: str,
    model_name: str,
    backbone: str,
    fold: int,
    thresholds: list[float],
    base_dir: str,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[float, list[dict[str, float]]]:
    """Run one fold's validation split and score every threshold from cached logits."""

    loaders = create_dataloaders(
        dataset=dataset,
        augment="no",
        aug_multiplier=1,
        fold=fold,
        batch_size=batch_size,
        base_dir=base_dir,
        num_workers=0,
        include_metadata=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = loaders["validation"]

    model = get_model(
        device,
        model_name=model_name,
        backbone=backbone,
        in_channels=1,
        out_channels=1,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    per_threshold: dict[float, list[dict[str, float]]] = {t: [] for t in thresholds}

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc=f"fold {fold} [val]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = get_logits(model, images)

            # One forward pass, reused across the whole grid.
            logits = logits.float().detach().cpu()
            masks = masks.detach().cpu()

            for item in range(logits.shape[0]):
                item_logits = logits[item : item + 1]
                item_mask = masks[item : item + 1]
                for threshold in thresholds:
                    pred = get_postprocessed_mask(item_logits, threshold=threshold)
                    per_threshold[threshold].append(compute_binary_mask_metrics(pred, item_mask))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return per_threshold


def run_experiment(
    experiment_dir: Path,
    thresholds: list[float],
    folds: list[int],
    device: torch.device,
    base_dir_override: str | None,
) -> pd.DataFrame | None:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        print(f"[skip] {experiment_dir.name}: no config.json")
        return None

    config = json.loads(config_path.read_text())
    dataset = config["dataset"]
    model_name = config["model_name"]
    backbone = config["backbone"]
    shipped_threshold = config.get("threshold")
    base_dir = base_dir_override or config.get("base_dir", ".")
    batch_size = int(config.get("batch_size", 8))
    use_amp = bool(config.get("use_amp", True))

    print(f"\n{'=' * 78}")
    print(f"{experiment_dir.name}  [{dataset} / {model_name} / {backbone}]")
    print(f"shipped threshold: {shipped_threshold}   grid: {thresholds}")
    print(f"{'=' * 78}")

    # fold -> threshold -> mean metric over that fold's validation images
    fold_means: dict[int, dict[float, dict[str, float]]] = {}

    for fold in folds:
        weights_path = experiment_dir / "weights" / f"{fold}.pth"
        if not weights_path.exists():
            print(f"[skip] fold {fold}: {weights_path} missing")
            continue

        per_threshold = sweep_fold(
            weights_path=weights_path,
            dataset=dataset,
            model_name=model_name,
            backbone=backbone,
            fold=fold,
            thresholds=thresholds,
            base_dir=base_dir,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
        )

        fold_means[fold] = {
            threshold: {m: float(np.mean([row[m] for row in rows])) for m in METRICS}
            for threshold, rows in per_threshold.items()
        }
        best_here = max(fold_means[fold], key=lambda t: fold_means[fold][t]["dice"])
        print(f"  fold {fold}: best val Dice at threshold {best_here:.2f} "
              f"({fold_means[fold][best_here]['dice']:.4f})")

    if not fold_means:
        print("[skip] no folds evaluated")
        return None

    # Fold-level averages -> across-fold mean/std, matching the paper's protocol.
    rows = []
    for threshold in thresholds:
        row: dict[str, float] = {"threshold": threshold, "n_folds": len(fold_means)}
        for metric in METRICS:
            values = [fold_means[f][threshold][metric] for f in fold_means]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)

    frame = pd.DataFrame(rows)
    best = frame.loc[frame["dice_mean"].idxmax()]

    print()
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"  selected threshold (max mean validation Dice): {best['threshold']:.2f}")
    print(f"  val Dice {best['dice_mean']:.4f} +/- {best['dice_std']:.4f}")
    if shipped_threshold is not None:
        shipped_row = frame[np.isclose(frame["threshold"], float(shipped_threshold))]
        if shipped_row.empty:
            print(f"  shipped threshold {shipped_threshold} is not in the grid")
        else:
            shipped_dice = float(shipped_row.iloc[0]["dice_mean"])
            delta = float(best["dice_mean"]) - shipped_dice
            verdict = "CONFIRMED" if np.isclose(best["threshold"], float(shipped_threshold)) else "DIFFERS"
            print(f"  shipped threshold {shipped_threshold}: val Dice {shipped_dice:.4f} "
                  f"(gap {delta:+.4f})  -> {verdict}")

    out_dir = experiment_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "threshold_sweep_validation.csv", index=False)
    (out_dir / "threshold_sweep_validation.json").write_text(
        json.dumps(
            {
                "experiment": experiment_dir.name,
                "dataset": dataset,
                "model_name": model_name,
                "backbone": backbone,
                "shipped_threshold": shipped_threshold,
                "selected_threshold": float(best["threshold"]),
                "split": "validation",
                "folds": sorted(fold_means),
                "per_fold_means": {str(f): {str(t): v for t, v in d.items()} for f, d in fold_means.items()},
                "summary": frame.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"  wrote {out_dir / 'threshold_sweep_validation.csv'}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Directory name under model/, e.g. BEST_tu_unet_effib3_t0.35")
    group.add_argument("--all-best", action="store_true", help="Sweep every model/BEST_* directory")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--thresholds", default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS))
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--base-dir", default=None, help="Override base_dir from config.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    folds = [int(part) for part in args.folds.split(",") if part.strip()]
    device = torch.device(args.device)
    model_root = Path(args.model_root)

    if args.all_best:
        targets = sorted(p for p in model_root.glob("BEST_*") if p.is_dir())
    else:
        targets = [model_root / args.experiment]

    if not targets:
        raise SystemExit(f"no experiment directories found under {model_root}")

    print("Threshold selection on the VALIDATION split. Test folds are never read.")

    for experiment_dir in targets:
        if not experiment_dir.exists():
            raise SystemExit(f"{experiment_dir} does not exist")
        run_experiment(experiment_dir, thresholds, folds, device, args.base_dir)


if __name__ == "__main__":
    main()
