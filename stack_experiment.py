"""Quick honest stacking experiment: XGBoost meta-learner on trained CNN outputs.

Tests whether feeding one or more trained segmentation models' per-pixel outputs into an
XGBoost meta-learner improves segmentation, evaluated with out-of-fold (OOF) predictions so
the result is trustworthy on this small dataset.

Two levels of cross-validation, both leakage-free:
  * Base level: in fold n, weights/n.pth never trained on fold n's images, so running each
    fold's test images through that fold's weights gives an OOF probability map for every image.
  * Meta level: the XGBoost meta-learner is cross-validated over the same image folds, so it
    never trains on the images it is evaluated on.

Pass one --base-experiment for a single-model refiner, or several for a multi-model blend.

Run example:
    python stack_experiment.py --base-experiment tu_unet_efficientnetb3 --dataset tu

    python stack_experiment.py --dataset tu \
        --base-experiment tu_unet_efficientnetb3 \
        --base-experiment tu_unetplusplus_efficientnetb3 \
        --base-experiment tu_deeplabv3plus_efficientnetb3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from tqdm.auto import tqdm

from utils.dataloader import create_dataloaders, get_kfold_chunks, normalize_dataset_name
from utils.metrics import compute_binary_mask_metrics, get_postprocessed_mask
from utils.model_factory import get_model, normalize_backbone_name, normalize_model_name
from utils.xgboost_segmenter import extract_pixel_features


METRIC_COLUMNS = ("dice", "iou", "precision", "recall", "specificity", "accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-experiment",
        action="append",
        required=True,
        dest="base_experiments",
        help="Trained base experiment folder name under --model-root. Repeat for a blend.",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset; defaults to the base configs' dataset.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--infer-batch-size", type=int, default=2, help="Mini-batch for base-model inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Used identically for base and stacked masks.")
    parser.add_argument("--window", type=int, default=9, help="Window for local mean/std of base probabilities.")
    parser.add_argument("--max-bg-ratio", type=int, default=5, help="Background pixels kept per foreground pixel.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rebuild-oof", action="store_true", help="Regenerate cached OOF outputs even if present.")
    return parser.parse_args()


def get_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        if len(output) == 1:
            return next(iter(output.values()))
        raise KeyError(f"Model output must contain 'out'. Keys: {', '.join(sorted(output))}")
    return output


def load_base_config(model_root: Path, experiment: str) -> dict:
    config_path = model_root / experiment / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing base config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def local_mean_std(prob: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mean = uniform_filter(prob, size=window, mode="reflect")
    mean_sq = uniform_filter(prob * prob, size=window, mode="reflect")
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
    return mean, std


# ----------------------------------------------------------------------------------------------
# Step A: generate & cache OOF base outputs
# ----------------------------------------------------------------------------------------------


def oof_cache_path(run_dir: Path, dataset: str, sample_id: int) -> Path:
    return run_dir / "oof" / f"{dataset}_{sample_id}.npz"


def generate_oof_cache(args: argparse.Namespace, run_dir: Path, dataset: str, bases: list[dict], device: torch.device) -> None:
    """Run each base model on its OOF test fold and cache image/mask/logits per sample."""

    (run_dir / "oof").mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)

    for fold in range(1, args.k_folds + 1):
        loaders = create_dataloaders(
            dataset=dataset,
            augment=False,
            aug_multiplier=1,
            fold=fold,
            batch_size=args.infer_batch_size,
            base_dir=args.base_dir,
            num_workers=0,
            include_metadata=True,
            pin_memory=device.type == "cuda",
        )

        # Collect this fold's test images/masks/ids once; all bases share the same split.
        images_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        sample_ids: list[int] = []
        for batch in loaders["test"]:
            for i in range(batch["image"].shape[0]):
                images_list.append(batch["image"][i, 0].numpy().astype(np.float32))
                masks_list.append(batch["mask"][i, 0].numpy().astype(np.float32))
                sample_ids.append(int(batch["sample_id"][i]))

        images_tensor = torch.from_numpy(np.stack(images_list)).unsqueeze(1)  # [N,1,H,W]

        # logits[base_index][sample_index] -> [H,W]
        base_logits: list[list[np.ndarray]] = []
        for base in bases:
            weights_path = model_root / base["experiment"] / "weights" / f"{fold}.pth"
            if not weights_path.exists():
                raise FileNotFoundError(f"Missing base weights: {weights_path}")
            model = get_model(
                device,
                model_name=base["model_name"],
                backbone=base["backbone"],
                in_channels=1,
                out_channels=1,
            )
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()

            per_sample: list[np.ndarray] = []
            with torch.no_grad():
                for start in tqdm(
                    range(0, images_tensor.shape[0], args.infer_batch_size),
                    desc=f"OOF fold {fold} [{base['experiment']}]",
                    leave=False,
                ):
                    chunk = images_tensor[start : start + args.infer_batch_size].to(device)
                    logits = get_logits(model, chunk).float().cpu().numpy()
                    for j in range(logits.shape[0]):
                        per_sample.append(logits[j, 0])
            base_logits.append(per_sample)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for idx, sample_id in enumerate(sample_ids):
            logits_stack = np.stack([base_logits[b][idx] for b in range(len(bases))]).astype(np.float32)
            np.savez_compressed(
                oof_cache_path(run_dir, dataset, sample_id),
                image=images_list[idx],
                mask=masks_list[idx],
                logits=logits_stack,
            )

    print(f"Cached OOF outputs for {args.k_folds} folds -> {run_dir / 'oof'}")


# ----------------------------------------------------------------------------------------------
# Step B: per-pixel meta-features
# ----------------------------------------------------------------------------------------------


def build_pixel_features(image: np.ndarray, base_logits: np.ndarray, window: int) -> np.ndarray:
    """Return [H*W, F] features: per-base logit/prob/local-mean/local-std + raw-image features."""

    feature_channels: list[np.ndarray] = []
    for logit in base_logits:  # [H,W] per base
        prob = sigmoid(logit)
        mean, std = local_mean_std(prob, window)
        feature_channels.extend([logit, prob, mean, std])

    stacked = np.stack(feature_channels, axis=-1).reshape(-1, len(feature_channels)).astype(np.float32)
    raw_features = extract_pixel_features(image)  # [H*W, 6]
    return np.concatenate([stacked, raw_features], axis=1)


# ----------------------------------------------------------------------------------------------
# Step C + D: meta cross-validation, metrics, comparison
# ----------------------------------------------------------------------------------------------


def mask_metrics_from_logit(logit_map: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    logit_tensor = torch.from_numpy(logit_map).unsqueeze(0).unsqueeze(0).float()
    pred = get_postprocessed_mask(logit_tensor, threshold=threshold)
    gt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    return compute_binary_mask_metrics(pred, gt)


def prob_to_logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(prob / (1.0 - prob)).astype(np.float32)


def fold_mean(per_image: list[dict[str, float]]) -> dict[str, float]:
    return {metric: float(np.mean([item[metric] for item in per_image])) for metric in METRIC_COLUMNS}


def summarize(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRIC_COLUMNS:
        values = [fold[metric] for fold in fold_metrics]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def evaluate_baseline(run_dir: Path, dataset: str, chunks: list[list[int]], bases: list[dict], threshold: float) -> dict:
    """Recompute each base model's OOF metrics from cached logits with identical postprocessing."""

    results: dict[str, dict] = {}
    for base_index, base in enumerate(bases):
        fold_metrics: list[dict[str, float]] = []
        for chunk in chunks:
            per_image: list[dict[str, float]] = []
            for sample_id in chunk:
                data = np.load(oof_cache_path(run_dir, dataset, sample_id))
                per_image.append(
                    mask_metrics_from_logit(data["logits"][base_index], data["mask"], threshold)
                )
            fold_metrics.append(fold_mean(per_image))
        results[base["experiment"]] = {
            "per_fold": fold_metrics,
            "summary": summarize(fold_metrics),
        }
    return results


def run_meta_cv(args: argparse.Namespace, run_dir: Path, dataset: str, chunks: list[list[int]], bases: list[dict]) -> dict:
    import xgboost as xgb

    rng = np.random.default_rng(args.seed)
    fold_metrics: list[dict[str, float]] = []

    for held_out_index, test_ids in enumerate(chunks):
        train_ids = [sid for i, chunk in enumerate(chunks) if i != held_out_index for sid in chunk]

        # Build subsampled training pixels from out-of-fold base outputs of the train images.
        x_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        for sample_id in train_ids:
            data = np.load(oof_cache_path(run_dir, dataset, sample_id))
            features = build_pixel_features(data["image"], data["logits"], args.window)
            labels = (data["mask"].ravel() >= 0.5).astype(np.int8)

            fg_idx = np.flatnonzero(labels == 1)
            bg_idx = np.flatnonzero(labels == 0)
            n_bg_keep = min(len(bg_idx), args.max_bg_ratio * max(len(fg_idx), 1))
            if n_bg_keep < len(bg_idx):
                bg_idx = rng.choice(bg_idx, size=n_bg_keep, replace=False)
            keep = np.concatenate([fg_idx, bg_idx])
            x_blocks.append(features[keep])
            y_blocks.append(labels[keep])

        x_train = np.concatenate(x_blocks)
        y_train = np.concatenate(y_blocks)
        n_pos = int(y_train.sum())
        scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)

        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=-1,
            random_state=args.seed,
            eval_metric="logloss",
        )
        print(
            f"Meta-fold {held_out_index + 1}/{len(chunks)}: train on {len(train_ids)} images "
            f"({len(x_train):,} pixels), test on {len(test_ids)} images"
        )
        clf.fit(x_train, y_train)

        per_image: list[dict[str, float]] = []
        for sample_id in test_ids:
            data = np.load(oof_cache_path(run_dir, dataset, sample_id))
            features = build_pixel_features(data["image"], data["logits"], args.window)
            proba = clf.predict_proba(features)
            fg_prob = proba[:, 1] if proba.shape[1] > 1 else np.full(features.shape[0], float(clf.classes_[0]))
            height, width = data["mask"].shape
            logit_map = prob_to_logit(fg_prob.reshape(height, width))
            per_image.append(mask_metrics_from_logit(logit_map, data["mask"], args.threshold))
        fold_metrics.append(fold_mean(per_image))

    return {"per_fold": fold_metrics, "summary": summarize(fold_metrics)}


def write_comparison(run_dir: Path, payload: dict) -> None:
    (run_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for name, result in payload["baselines"].items():
        rows.append({"model": f"base:{name}", **{m: result["summary"][m]["mean"] for m in METRIC_COLUMNS}})
    rows.append({"model": "stacked", **{m: payload["stacked"]["summary"][m]["mean"] for m in METRIC_COLUMNS}})

    csv_path = run_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(payload: dict) -> None:
    print("\n" + "=" * 64)
    print("Stacking experiment summary (mean Dice / IoU across folds)")
    print("=" * 64)
    best_base_name, best_base_dice = None, -1.0
    for name, result in payload["baselines"].items():
        dice = result["summary"]["dice"]["mean"]
        iou = result["summary"]["iou"]["mean"]
        print(f"  base   {name:<45} Dice={dice:.4f}  IoU={iou:.4f}")
        if dice > best_base_dice:
            best_base_name, best_base_dice = name, dice

    stacked = payload["stacked"]["summary"]
    stacked_dice = stacked["dice"]["mean"]
    stacked_std = stacked["dice"]["std"]
    print(f"  STACK  {'(' + str(len(payload['bases'])) + ' base model(s))':<45} "
          f"Dice={stacked_dice:.4f}  IoU={stacked['iou']['mean']:.4f}")
    print("-" * 64)

    delta = stacked_dice - best_base_dice
    verdict = (
        "IMPROVES over best base" if delta > stacked_std
        else "does NOT clearly beat best base (within ~1 std)"
    )
    print(f"  Best base: {best_base_name} (Dice={best_base_dice:.4f})")
    print(f"  Stacked - best base Dice delta = {delta:+.4f}  (stacked fold std={stacked_std:.4f})")
    print(f"  Verdict: stacking {verdict}.")
    print("=" * 64)


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve base model configs and the shared dataset.
    bases: list[dict] = []
    datasets: set[str] = set()
    for experiment in args.base_experiments:
        config = load_base_config(model_root, experiment)
        bases.append({
            "experiment": experiment,
            "model_name": normalize_model_name(config.get("model_name", "unet")),
            "backbone": normalize_backbone_name(config.get("backbone", "efficientnet-b3")),
        })
        datasets.add(normalize_dataset_name(config.get("dataset", "tu")))

    if args.dataset is not None:
        dataset = normalize_dataset_name(args.dataset)
    elif len(datasets) == 1:
        dataset = datasets.pop()
    else:
        raise ValueError(f"Base experiments span multiple datasets {datasets}; pass --dataset to disambiguate.")

    run_name = args.run_name or ("stack_" + "_AND_".join(args.base_experiments))
    run_dir = model_root / "stack_experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(f"Base models: {[b['experiment'] for b in bases]}")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")

    chunks = get_kfold_chunks(dataset)[: args.k_folds]

    # Step A: cache OOF outputs (skip if already present unless --rebuild-oof).
    first_sample = chunks[0][0]
    if args.rebuild_oof or not oof_cache_path(run_dir, dataset, first_sample).exists():
        generate_oof_cache(args, run_dir, dataset, bases, device)
    else:
        print("Reusing existing OOF cache (pass --rebuild-oof to regenerate).")

    # Step D baseline + Step C stacked.
    baselines = evaluate_baseline(run_dir, dataset, chunks, bases, args.threshold)
    stacked = run_meta_cv(args, run_dir, dataset, chunks, bases)

    payload = {
        "dataset": dataset,
        "bases": bases,
        "threshold": args.threshold,
        "window": args.window,
        "max_bg_ratio": args.max_bg_ratio,
        "xgboost": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        },
        "baselines": baselines,
        "stacked": stacked,
    }
    write_comparison(run_dir, payload)
    print_summary(payload)
    print(f"\nSaved comparison -> {run_dir / 'comparison.json'} and comparison.csv")


if __name__ == "__main__":
    main()
