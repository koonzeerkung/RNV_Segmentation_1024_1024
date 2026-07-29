"""
Recover exact per-image confusion matrices from the saved per-image metrics and
report pixel-pooled (micro) precision / recall / F1 for the six paper experiments.

Why this exists
---------------
Everything the pipeline reports today is a *macro* average: precision, recall and
Dice are computed per image, then averaged over images and folds. F1 is not linear,
so the reported F1 (= mean of per-image Dice) is not the harmonic mean of the
reported mean precision and mean recall. The gap runs +0.03..+0.07 on these
experiments, which is exactly the kind of arithmetic a reviewer checks.

Micro averaging fixes the identity: pool tp / fp / fn over all test images, then
compute precision, recall and F1 once. By construction

    2 * P * R / (P + R)  ==  2 * tp / (2 * tp + fp + fn)

holds to floating point.

No re-run is required. `outputs*/test_fold_<n>/per_image_metrics.csv` stores float32
precision, and precision + recall + accuracy determine tp, fp, fn exactly for a known
image size:

    fp = tp * (1 - P) / P
    fn = tp * (1 - R) / R
    tp + tn = accuracy * N   with   tn = N - tp - fp - fn
    =>  tp = N * (1 - accuracy) / ((1 - P) / P + (1 - R) / R)

Recovered counts come back as integers and reproduce the stored Dice to ~1e-8; the
script verifies this per image and refuses to report a row whose reconstruction drifts.

Usage:
    python compute_micro_metrics.py
    python compute_micro_metrics.py --output micro_metrics.csv --image-size 1024
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

import pandas as pd

EXPERIMENTS = [
    ("tu", "U-Net", "BEST_tu_unet_effib3_t0.35"),
    ("tu", "U-Net++", "BEST_tu_unetplusplus_effib3_t0.35"),
    ("tu", "DeepLabv3+", "BEST_tu_deeplabv3plus_efficientnetb3"),
    ("drac", "U-Net", "BEST_drac_unet_effib3_t0.35"),
    ("drac", "U-Net++", "BEST_drac_unetplusplus_effib3_t0.35"),
    ("drac", "DeepLabv3+", "BEST_drac_deeplabv3plus_efficientnetb3"),
]

STAGES = [("no_filter", "outputs"), ("filter", "outputs_nearby")]

# Below this, precision/recall are dominated by the smooth term rather than by tp.
DEGENERATE = 1e-4
# Reconstructed Dice must match the stored Dice this closely.
RECON_TOLERANCE = 1e-5

FOLD_RE = re.compile(r"test_fold_(\d+)")


class ReconstructionError(Exception):
    """Raised when a per-image row cannot be turned back into a confusion matrix."""


def recover_counts(row: pd.Series, n_pixels: int) -> tuple[float, float, float, float]:
    """Return (tp, fp, fn, tn) for one image, in pixels."""

    precision = float(row["precision"])
    recall = float(row["recall"])
    accuracy = float(row["accuracy"])
    specificity = float(row["specificity"])

    if precision < DEGENERATE or recall < DEGENERATE:
        # tp is effectively zero, so the precision/recall route carries no scale.
        # Fall back on accuracy + specificity, which still pin down fp and fn.
        tp = 0.0
        tn = accuracy * n_pixels
        fp = tn * (1.0 - specificity) / specificity if specificity > 0 else 0.0
        fn = n_pixels - tn - fp
        return tp, max(fp, 0.0), max(fn, 0.0), tn

    if accuracy >= 1.0:
        # A pixel-perfect prediction leaves every ratio at 1.0 and destroys the scale.
        raise ReconstructionError("accuracy == 1.0, tp is unrecoverable from ratios")

    denom = (1.0 - precision) / precision + (1.0 - recall) / recall
    if denom <= 0:
        raise ReconstructionError("precision == recall == 1.0 with accuracy < 1.0")

    tp = n_pixels * (1.0 - accuracy) / denom
    fp = tp * (1.0 - precision) / precision
    fn = tp * (1.0 - recall) / recall
    tn = n_pixels - tp - fp - fn
    return tp, fp, fn, tn


def verify(row: pd.Series, tp: float, fp: float, fn: float) -> float:
    """Return the absolute Dice reconstruction error for one image."""

    stored = float(row["dice"])
    total = 2.0 * tp + fp + fn
    rebuilt = (2.0 * tp / total) if total > 0 else 0.0
    return abs(stored - rebuilt)


def micro_from_counts(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_direct = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f1_counts = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1_direct,
        "micro_f1_from_counts": f1_counts,
        "micro_iou": iou,
        "identity_gap": abs(f1_direct - f1_counts),
    }


def macro_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    macro_p = float(frame["precision"].mean())
    macro_r = float(frame["recall"].mean())
    harmonic = 2.0 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    macro_f1 = float(frame["dice"].mean())
    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_hm_of_means": harmonic,
        "macro_gap": harmonic - macro_f1,
    }


def load_stage(stage_dir: Path, n_pixels: int) -> tuple[pd.DataFrame, list[str]]:
    """Load every per-image row under a stage directory, with counts attached."""

    files = sorted(
        glob.glob(str(stage_dir / "test_fold_*" / "per_image_metrics.csv")),
        key=lambda path: int(FOLD_RE.search(path).group(1)),
    )
    if not files:
        return pd.DataFrame(), [f"no per_image_metrics.csv under {stage_dir}"]

    warnings: list[str] = []
    frames = []
    for path in files:
        fold = int(FOLD_RE.search(path).group(1))
        frame = pd.read_csv(path)
        counts = []
        for _, row in frame.iterrows():
            try:
                tp, fp, fn, tn = recover_counts(row, n_pixels)
            except ReconstructionError as exc:
                warnings.append(f"fold {fold}, sample {row.get('sample_id')}: {exc}")
                continue
            error = verify(row, tp, fp, fn)
            if error > RECON_TOLERANCE:
                warnings.append(
                    f"fold {fold}, sample {row.get('sample_id')}: "
                    f"dice reconstruction off by {error:.2e}"
                )
                continue
            counts.append(
                {
                    "fold": fold,
                    "sample_id": row.get("sample_id"),
                    "tp": round(tp),
                    "fp": round(fp),
                    "fn": round(fn),
                    "tn": round(tn),
                    "dice": float(row["dice"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "recon_error": error,
                }
            )
        frames.append(pd.DataFrame(counts))

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return combined, warnings


def summarize(frame: pd.DataFrame, scope: str) -> dict[str, float]:
    row: dict[str, float] = {
        "scope": scope,
        "n_images": int(len(frame)),
        "tp": int(frame["tp"].sum()),
        "fp": int(frame["fp"].sum()),
        "fn": int(frame["fn"].sum()),
        "max_recon_error": float(frame["recon_error"].max()),
    }
    row.update(micro_from_counts(row["tp"], row["fp"], row["fn"]))
    row.update(macro_from_frame(frame))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-root", default="model", help="root holding the experiment dirs")
    parser.add_argument("--image-size", type=int, default=1024, help="side length in pixels")
    parser.add_argument("--output", default="micro_metrics.csv", help="where to write the table")
    parser.add_argument(
        "--per-fold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include one row per fold alongside the pooled row",
    )
    args = parser.parse_args()

    n_pixels = args.image_size * args.image_size
    model_root = Path(args.model_root)

    rows: list[dict] = []
    all_warnings: list[str] = []

    for dataset, model, exp_name in EXPERIMENTS:
        for stage_label, stage_subdir in STAGES:
            stage_dir = model_root / exp_name / stage_subdir
            frame, warnings = load_stage(stage_dir, n_pixels)
            all_warnings += [f"{exp_name}/{stage_subdir}: {w}" for w in warnings]
            if frame.empty:
                continue

            scopes = []
            if args.per_fold:
                for fold in sorted(frame["fold"].unique()):
                    scopes.append((f"fold_{int(fold)}", frame[frame["fold"] == fold]))
            scopes.append(("pooled", frame))

            for scope, subset in scopes:
                row = {"dataset": dataset, "model": model, "experiment": exp_name,
                       "stage": stage_label}
                row.update(summarize(subset, scope))
                rows.append(row)

    if not rows:
        print("No per-image metrics found. Run visualize_nearby.py first.")
        return

    table = pd.DataFrame(rows)
    column_order = [
        "dataset", "model", "experiment", "stage", "scope", "n_images",
        "tp", "fp", "fn",
        "micro_precision", "micro_recall", "micro_f1", "micro_f1_from_counts",
        "micro_iou", "identity_gap",
        "macro_precision", "macro_recall", "macro_f1", "macro_hm_of_means", "macro_gap",
        "max_recon_error",
    ]
    table = table[column_order]
    table.to_csv(args.output, index=False)

    pooled = table[table["scope"] == "pooled"]
    print("Pixel-pooled (micro) vs per-image mean (macro)\n")
    header = (
        f"{'dataset':>6} {'model':>11} {'stage':>10} "
        f"{'microP':>7} {'microR':>7} {'microF1':>8} | "
        f"{'macroP':>7} {'macroR':>7} {'macroF1':>8} {'HM(macro)':>10} {'gap':>7}"
    )
    print(header)
    print("-" * len(header))
    for _, row in pooled.iterrows():
        print(
            f"{row['dataset']:>6} {row['model']:>11} {row['stage']:>10} "
            f"{row['micro_precision']:7.4f} {row['micro_recall']:7.4f} {row['micro_f1']:8.4f} | "
            f"{row['macro_precision']:7.4f} {row['macro_recall']:7.4f} {row['macro_f1']:8.4f} "
            f"{row['macro_hm_of_means']:10.4f} {row['macro_gap']:+7.4f}"
        )

    worst_identity = pooled["identity_gap"].max()
    worst_recon = pooled["max_recon_error"].max()
    print(
        f"\nmicro F1 identity holds to {worst_identity:.2e}; "
        f"worst confusion-matrix reconstruction error {worst_recon:.2e}"
    )
    print(f"Wrote {len(table)} rows to {args.output}")

    if all_warnings:
        print(f"\n{len(all_warnings)} image(s) skipped:")
        for warning in all_warnings[:20]:
            print(f"  {warning}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more")


if __name__ == "__main__":
    main()
