# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A 5-fold cross-validation segmentation pipeline for retinal vessel images. Two datasets (`dataset_tu`, `dataset_drac`) containing 1024×1024 grayscale images. Models are trained with `main.py` and evaluated/visualized with `visualize_nearby.py`. Results are saved under `model/<experiment_name>/`.

## Common Commands

```bash
# Train a single experiment (reads [train] section from TOML)
./run_main.sh --config configs/my_experiment.toml

# Evaluate and visualize after training
./run_visualize.sh --config configs/my_experiment.toml --experiment-name <name>

# Preview the full architecture × backbone × dataset sweep without running
./run_experiment_queue.sh --dry-run

# Run the full queue (deeplabv3plus, unetplusplus, unet × 3 backbones × 2 datasets)
./run_experiment_queue.sh

# Queue with extra training overrides passed through
./run_experiment_queue.sh --train-arg=--epochs --train-arg=50

# Cross-dataset summary for the same model setting on both datasets
./run_visualize.sh --config configs/my_experiment.toml --summary-only --summarize-all-datasets

# Single-image inspection with one specific weight file
python specify_visualize.py model/<experiment_name>/weights/1.pth tu 1
```

All shell scripts (`run_main.sh`, `run_visualize.sh`, `run_experiment_queue.sh`) optionally activate a conda environment if `PIPELINE_CONDA_ENV` is set, then delegate to the corresponding Python script. CLI arguments override the TOML config.

## Config System

All tuneable values live in TOML files under `configs/`. Two sections:

- `[train]` — read by `main.py`
- `[visualize]` — read by `visualize_nearby.py`

`utils/config.py::dataclass_from_section()` merges the TOML section over the script's frozen dataclass defaults. Unknown keys raise an error. The typical workflow is to copy `configs/default.toml`, edit the copy, and pass it with `--config`.

Leave `experiment_name = ""` in TOML to auto-generate a name from the full config (dataset, model, aug settings, lr, etc.). Set a fixed name to avoid the long auto-generated slug.

## Architecture

### Data path
`utils/dataloader.py::create_dataloaders()` → `build_samples()` (resolves file paths for each fold) → `filter_samples_with_answers()` (drops training images with empty masks) → `KFoldSliceDataset`.

`KFoldSliceDataset.__init__` pre-loads all unique images and masks into RAM as uint8 numpy arrays so `__getitem__` never touches disk. The dataset is small enough (27–42 images) for this to cost well under 200 MB. Augmentation (HorizontalFlip, VerticalFlip, Affine, CLAHE) is applied per-item inside `__getitem__` only for the training split when `augment_index > 0`.

Fold layout is fixed:
- `dataset_tu`: folds cover sample IDs 1–6, 7–12, 13–17, 18–22, 23–27
- `dataset_drac`: folds cover 1–9, 10–18, 19–26, 27–34, 35–42
- For fold `n`: test = fold `n`, validation = fold `n+1` (wraps), train = remaining folds

### Model path
`utils/model_factory.py::get_model()` builds a `segmentation_models_pytorch` model and wraps it in `WrappedModel`, which always returns `{"out": logits}`. Both `main.py` and `visualize_nearby.py` use an identical `get_logits()` helper that extracts `output["out"]` from the dict (or returns the tensor directly if the model doesn't return a dict).

Supported architectures: `unet`, `unetplusplus`, `deeplabv3plus`, `deeplabv3`.  
Supported backbones: `efficientnet-b3`, `inceptionv4`, `densenet169`.  
**Constraint**: DeepLab architectures only work with `efficientnet-b3`. Using `inceptionv4` or `densenet169` with DeepLab raises `ValueError` at model creation.

### Loss and metrics path
`utils/metrics.py`:
- Training uses `BCEDiceLoss` (BCE + soft Dice on raw logits) and `dice_score_from_logits` for per-batch feedback (no postprocessing).
- Validation uses `compute_binary_metrics(logits, masks, postprocess=True)`, which calls `get_postprocessed_mask` (sigmoid → threshold → `scipy.ndimage.binary_fill_holes`). The postprocessed mask drives best-model saving, early stopping, and the scheduler.
- Hole-filling transfers a bool tensor (not float32) from GPU to CPU to minimize transfer size.
- `compute_binary_mask_metrics` is available when a postprocessed mask is already in hand (avoids re-running sigmoid/threshold/hole-fill).

### Training loop (`main.py`)
`run_fold()` handles one fold: creates dataloaders, builds model + Adam optimizer + `ReduceLROnPlateau` scheduler, then runs the epoch loop. AMP and gradient accumulation are on by default. Best model is saved by whichever `best_metric` is configured (`dice` by default). Early stopping fires after `patience` epochs with no improvement. History (loss, Dice, LR, elapsed times) is saved as JSON and PNG per fold.

### Evaluation loop (`visualize_nearby.py`)
Loads each fold's `.pth` weights, rebuilds the test dataloader, runs inference at batch_size=1 (default), and writes visualization panels and per-image metrics to `outputs/test_fold_<fold>/`. Optional representative-point filtering (from a CSV of x,y coordinates) keeps only predicted blobs within `radius_tolerance` pixels of a known location; filtered results write to `outputs_nearby_<points_mode>/` without overwriting normal results. A summary CSV/JSON across all folds is always written.

`points_mode` selects which point set the filter uses, and hence the output directory: `multiple` (default) uses every point the localizer emits, several per lesion; `single` uses one clustered representative per lesion. The CSV for each mode/dataset pair is resolved in `utils/dataset_points.py::resolve_points_csv()`; setting `points_csv` explicitly overrides the file but not the directory.

### Experiment queue (`run_experiment_queue.py`)
Iterates all requested dataset × model × backbone triples, skips invalid DeepLab combinations, skips already-completed experiments by default (checks for `test_metrics.json`), and serializes queue state to `model/experiment_queue_state.json`. Use `--continue-on-error` to keep going after failures.

## Output Layout

```
model/
  <experiment_name>/
    config.json                        # resolved config at run time
    weights/<fold>.pth                 # best model per fold
    outputs/
      loss_dice_fold_<fold>.png
      history_fold_<fold>.json
      best_model_timing_fold_<fold>.json
      kfold_summary.json
      test_fold_<fold>/                # visualization panels per image
      test_metrics.csv / .json
      test_metrics_summary.csv / .json
    outputs_nearby_multiple/           # same structure, filtered with all localizer points
    outputs_nearby_single/             # same structure, filtered with one point per lesion
  dataset_summaries/
    <setting_suffix>.csv / .json       # combined tu+drac summary
  experiment_queue_state.json
  experiment_queue_logs/
```

## Adding a New Model

1. Add the model class and its aliases to `_MODEL_ALIASES` and `_ARCHITECTURE_CLASSES` in `utils/model_factory.py`.
2. Update `validate_model_backbone()` if certain backbone combinations are invalid.
3. If the new model needs a non-standard training loop (e.g., no gradient descent), add a separate `run_fold_<model>()` function in `main.py` and dispatch to it from `main()` based on `args.model_name`.
4. Ensure `forward()` returns either a plain tensor or a dict with an `"out"` key — `get_logits()` handles both.
5. If `state_dict()` / `load_state_dict()` need to be overridden (e.g., for non-PyTorch models), override both since `torch.save` / `torch.load` use pickle internally and can handle non-tensor dicts.

## Key Constraints

- **Do not edit `main.py` or `visualize_nearby.py` for normal experiments** — use TOML config and CLI overrides instead.
- `tomllib` requires Python 3.11+. On older Python, `tomli` is used as a fallback in `run_experiment_queue.py` only.
- The pipeline has no test suite. Correctness is verified by running training and checking saved metrics.
- `visualize_nearby.py` reads `model_name` and `backbone` from the saved `config.json` when available, overriding what is in the TOML `[visualize]` section.
