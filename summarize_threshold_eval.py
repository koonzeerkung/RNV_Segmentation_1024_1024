"""
Consolidate the no-filter vs candidate-point-filter test results of the six paper
experiments into one table, and check the invariant the filter must satisfy.

Reads, for each experiment:
    model/<exp>/outputs/test_metrics_summary.json                   (no filter)
    model/<exp>/outputs_nearby_<mode>/test_metrics_summary.json     (candidate-point filter)
    model/<exp>/outputs*/test_metrics.csv                           (per-fold, for the invariant)

<mode> is "multiple" (every localizer point) or "single" (one representative per lesion),
selected with --points-mode. Results produced before points_mode existed sit in a plain
outputs_nearby/ directory and are used as a fallback for --points-mode multiple.

Invariant: the filter only deletes connected components, so per fold it can raise
precision but must never raise recall. A fold where recall increases after filtering
means the two passes did not come from the same model state -- exactly the defect
recorded in ref/PAPER_TODO.md for DRAC DeepLabV3+.

Usage:
    python summarize_threshold_eval.py --threshold 0.45 --output summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# config toml is irrelevant here; only the experiment directory names matter.
EXPERIMENTS = [
    ("tu", "U-Net", "BEST_tu_unet_effib3_t0.35"),
    ("tu", "U-Net++", "BEST_tu_unetplusplus_effib3_t0.35"),
    ("tu", "DeepLabv3+", "BEST_tu_deeplabv3plus_efficientnetb3"),
    ("drac", "U-Net", "BEST_drac_unet_effib3_t0.35"),
    ("drac", "U-Net++", "BEST_drac_unetplusplus_effib3_t0.35"),
    ("drac", "DeepLabv3+", "BEST_drac_deeplabv3plus_efficientnetb3"),
]

METRICS = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]

# Recall may dip by a hair from float noise; anything beyond this is a real increase.
RECALL_TOLERANCE = 1e-6


def nearby_dir(exp_dir: Path, points_mode: str) -> Path:
    """Filtered-results directory for a mode, falling back to the pre-points_mode layout."""

    current = exp_dir / f"outputs_nearby_{points_mode}"
    legacy = exp_dir / "outputs_nearby"
    if not current.exists() and points_mode == "multiple" and legacy.exists():
        return legacy
    return current


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_folds(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def check_recall_invariant(exp_dir: Path, points_mode: str) -> list[str]:
    """Return a list of violation messages (empty if the filter behaved)."""

    plain = load_folds(exp_dir / "outputs" / "test_metrics.csv")
    filtered = load_folds(nearby_dir(exp_dir, points_mode) / "test_metrics.csv")
    if plain is None or filtered is None:
        return ["missing per-fold test_metrics.csv on one side"]

    merged = plain.merge(filtered, on="fold", suffixes=("_plain", "_filt"))
    if merged.empty:
        return ["no overlapping folds between the two passes"]

    violations = []
    for _, row in merged.iterrows():
        delta = row["recall_filt"] - row["recall_plain"]
        if delta > RECALL_TOLERANCE:
            violations.append(
                f"fold {int(row['fold'])}: recall rose {row['recall_plain']:.4f} "
                f"-> {row['recall_filt']:.4f} (+{delta:.4f})"
            )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", default="", help="Recorded in the output for provenance only")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--prefix", default="BEST_",
                        help="Experiment directory prefix, e.g. RETRAIN_ for the Drive-parameter retrain")
    parser.add_argument("--points-mode", choices=("multiple", "single"), default="multiple",
                        help="Which filtered-results directory to read (outputs_nearby_<mode>)")
    parser.add_argument("--output", default="threshold_eval_summary.csv")
    args = parser.parse_args()

    model_root = Path(args.model_root)
    rows: list[dict] = []
    problems: list[str] = []

    for dataset, model_label, exp in EXPERIMENTS:
        exp = args.prefix + exp.removeprefix("BEST_")
        exp_dir = model_root / exp
        filtered_dir = nearby_dir(exp_dir, args.points_mode)
        plain = load_summary(exp_dir / "outputs" / "test_metrics_summary.json")
        filtered = load_summary(filtered_dir / "test_metrics_summary.json")

        if plain is None:
            problems.append(f"{exp}: outputs/test_metrics_summary.json missing")
        if filtered is None:
            problems.append(f"{exp}: {filtered_dir.name}/test_metrics_summary.json missing")
        if plain is None or filtered is None:
            continue

        for label, summary in (("no filter", plain), ("filter", filtered)):
            row = {
                "dataset": dataset,
                "model": model_label,
                "experiment": exp,
                "stage": label,
                "points_mode": args.points_mode,
                "threshold": args.threshold,
                "n_folds": summary.get("num_folds"),
            }
            for metric in METRICS:
                row[metric] = summary["mean"].get(metric)
                row[f"{metric}_std"] = summary["std"].get(metric)
            rows.append(row)

        for message in check_recall_invariant(exp_dir, args.points_mode):
            problems.append(f"{exp}: {message}")

    if not rows:
        raise SystemExit("no results found -- did the evaluation run?")

    frame = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    frame.to_json(out_path.with_suffix(".json"), orient="records", indent=2)

    # Compact console view: one line per experiment, both stages side by side.
    print()
    print(f"threshold {args.threshold or '(unspecified)'}   points mode {args.points_mode}")
    print(f"{'dataset':<6} {'model':<11} {'stage':<10} "
          f"{'dice':>16} {'iou':>16} {'precision':>16} {'recall':>16}")
    print("-" * 96)
    for _, row in frame.iterrows():
        cells = " ".join(f"{row[m]:.4f}+/-{row[f'{m}_std']:.4f}" for m in
                         ["dice", "iou", "precision", "recall"])
        print(f"{row['dataset']:<6} {row['model']:<11} {row['stage']:<10} {cells}")

    # Filter deltas.
    print()
    print("effect of the candidate-point filter (filter - no filter)")
    print(f"{'dataset':<6} {'model':<11} {'d dice':>9} {'d iou':>9} "
          f"{'d precision':>12} {'d recall':>10}")
    print("-" * 62)
    for (dataset, model_label), group in frame.groupby(["dataset", "model"], sort=False):
        plain = group[group["stage"] == "no filter"]
        filt = group[group["stage"] == "filter"]
        if plain.empty or filt.empty:
            continue
        deltas = {m: float(filt.iloc[0][m]) - float(plain.iloc[0][m])
                  for m in ["dice", "iou", "precision", "recall"]}
        print(f"{dataset:<6} {model_label:<11} "
              f"{deltas['dice']:>+9.4f} {deltas['iou']:>+9.4f} "
              f"{deltas['precision']:>+12.4f} {deltas['recall']:>+10.4f}")

    print()
    if problems:
        print(f"INVARIANT / DATA PROBLEMS ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("A recall increase after filtering is impossible for a deletion-only "
              "filter; it means the two passes used different weights.")
    else:
        print("invariant OK: recall never increased after filtering, in any fold")

    print()
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
