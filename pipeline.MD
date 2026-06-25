# Performance Comparison Pipeline

## Sharing And Setup

For friends running this pipeline, use these files:

- `environment.yml`: optional conda environment template.
- `requirements.txt`: pip dependencies, useful for Google Colab.
- `configs/default.toml`: editable training and visualization config.
- `configs/colab.toml`: Google Colab example config.
- `run_main.sh`: runs training, optionally activating a conda env if `PIPELINE_CONDA_ENV` is set.
- `run_visualize.sh`: runs standalone evaluation/visualization, optionally activating a conda env if `PIPELINE_CONDA_ENV` is set.
- `specify_visualize.py`: visualizes one selected sample with one selected trained weight file.
- `SPECIFY_VISUALIZE.md`: single-sample visualization instructions.
- `README.md`: short quick-start instructions.
- `COLAB.md`: Colab-specific setup and run instructions.

Recommended workflow:

```bash
conda env create -f environment.yml
conda activate rnv
cp configs/default.toml configs/my_experiment.toml
bash run_main.sh --config configs/my_experiment.toml
bash run_visualize.sh --config configs/my_experiment.toml --experiment-name <experiment_name>
```

The environment name does not need to be `rnv`; it only needs the dependencies installed. The config file is the intended editing point. Avoid editing `main.py` or `visualize_nearby.py` for normal experiments.

For Google Colab:

```python
!pip install -r requirements.txt
!python main.py --config configs/colab.toml
```

Mount Google Drive and set `model_root` in `configs/colab.toml` if outputs should persist after the Colab runtime stops.

## `utils/dataloader.py`

Creates PyTorch dataloaders for one selected dataset using the fold rules from `command.md`.

Public entry point:

```python
from utils.dataloader import create_dataloaders

loaders = create_dataloaders(
    dataset="tu",          # "tu", "drac", "dataset_tu", or "dataset_drac"
    augment="yes",         # "yes"/"no" or bool
    aug_multiplier=2,      # applied to training samples only
    fold=1,                # 1-based fold number, valid range 1-5
    batch_size=8,          # default batch size
    base_dir=".",          # project directory containing dataset_tu/dataset_drac
    include_metadata=True, # set False for faster training batches
    pin_memory=False,      # set True when loading to CUDA
)
```

Returned object:

```python
{
    "train": train_loader,
    "validation": validation_loader,
    "test": test_loader,
}
```

### Dataset Selection

- `dataset="tu"` maps to `dataset_tu`
- `dataset="drac"` maps to `dataset_drac`

Both datasets must contain:

```text
dataset_name/
  raw/<sample_id>.<image extension>
  gt/<sample_id>.<image extension>
```

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.

### Fold Logic

The dataloader uses five fixed folds.

`dataset_tu`:

- Fold 1: samples 1-6
- Fold 2: samples 7-12
- Fold 3: samples 13-17
- Fold 4: samples 18-22
- Fold 5: samples 23-27

`dataset_drac`:

- Fold 1: samples 1-9
- Fold 2: samples 10-18
- Fold 3: samples 19-26
- Fold 4: samples 27-34
- Fold 5: samples 35-42

For a given `fold=n`:

- test set: fold `n`
- validation set: fold `n+1`
- train set: all remaining folds

Validation wraps around after fold 5, so `fold=5` uses fold 1 as validation.

### Image Processing

- Raw images are loaded as grayscale.
- Ground-truth masks are loaded as grayscale.
- Each 1024x1024 image/mask pair is used as one full-size sample.
- Returned `image` tensors are float tensors shaped `[1, 1024, 1024]`, normalized to `[0, 1]`.
- Returned `mask` tensors are float tensors shaped `[1, 1024, 1024]`, binarized with threshold `0.5`.

When `include_metadata=True`, each batch item also includes metadata:

- `dataset`
- `split`
- `sample_id`
- `patch_id` (kept for compatibility; always `0` for full-image samples)
- `augment_index`
- `augmentation`
- `image_path`
- `mask_path`

When `include_metadata=False`, batches contain only `image` and `mask`. `main.py` uses this faster mode during training and validation to reduce Python object collation and batch memory.

### Augmentation

Augmentation is applied only to the training split.

- `augment="no"` disables augmentation and ignores `aug_multiplier`.
- `augment="yes"` repeats each training image `aug_multiplier` times and applies randomized Albumentations transforms to repeated samples.
- `augment_index=0` is the original image.
- `augment_index>0` uses the training augmentation pipeline.
- Training images whose ground-truth mask contains no foreground pixels are removed before the augmentation multiplier is applied. Validation and test images are not filtered.

The training augmentation pipeline uses only the requested techniques:

- `A.HorizontalFlip(p=0.5)`
- `A.VerticalFlip(p=0.5)`
- `A.Affine(...)` for geometric transformations with rotate, scale, translate, and shear
- `A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5)`

Spatial augmentations are applied identically to the image and mask using bilinear interpolation for raw images and nearest-neighbor interpolation for masks. CLAHE is applied only to the raw image because the ground-truth mask must remain a label map.

## `utils/metrics.py`

Provides loss, mask postprocessing, and binary segmentation metrics.

Main components:

- `BCEDiceloss`: Combined BCE-with-logits and soft Dice loss for binary segmentation.
- `BCEDiceLoss`: Alias for `BCEDiceloss`.
- `dice_loss_from_logits(pred_logits, targets)`: Soft Dice loss from raw model logits.
- `get_postprocessed_mask(pred_logits, threshold=0.5)`: Converts logits to a binary mask and fills holes with `scipy.ndimage.binary_fill_holes`.
- `get_binary_mask(pred_logits, threshold=0.5)`: Converts logits to a binary mask without hole filling.
- `dice_coefficient(preds, targets)`: Dice score from binary masks.
- `dice_score_from_logits(pred_logits, targets)`: Lightweight Dice score from logits.
- `iou_score(preds, targets)`: IoU score from binary masks.
- `compute_binary_metrics(pred_logits, targets, threshold=0.5, postprocess=True)`: Returns Dice, IoU, precision, recall, specificity, and accuracy.
- `compute_binary_mask_metrics(preds, targets)`: Returns the same metrics from an already binary prediction mask.

Expected tensor shapes:

- `pred_logits`: `[B, 1, H, W]` raw logits from the model.
- `targets`: `[B, 1, H, W]` binary ground-truth masks.

Training should use `BCEDiceloss` directly on raw logits. Evaluation can use `compute_binary_metrics`; by default it applies the same hole-filling postprocessing requested in `command.md`.

Performance-related helpers:

- `dice_score_from_logits(...)`: Computes only Dice from logits for lightweight training feedback.
- `compute_binary_mask_metrics(...)`: Computes metrics from an already postprocessed binary mask, avoiding duplicate sigmoid/threshold/hole-filling work.

## `main.py`

Script version of the original `main.ipynb` training workflow.

It uses the project utility modules directly:

- `utils.dataloader.create_dataloaders`
- `utils.metrics.BCEDiceLoss`
- `utils.metrics.compute_binary_metrics`
- `utils.model_factory.get_model`

Default behavior:

- trains the dataset selected in the editable config
- trains the selected architecture/backbone pair, defaulting to `model_name="unet"` and `backbone="efficientnet-b3"`
- uses 5 folds
- uses augmentation with `aug_multiplier=10`
- trains for up to the configured number of epochs per fold
- uses early stopping with the configured patience
- saves each experiment under `model/<experiment_name>/`
- saves best model weights to `model/<experiment_name>/weights/<fold>.pth`
- saves loss/Dice plots and JSON histories to `model/<experiment_name>/outputs/`
- saves the resolved configuration to `model/<experiment_name>/config.json`
- lets `best_metric` choose the validation metric for best-model saving, early stopping, and scheduler plateau detection
- prints split sample counts for every fold:
  - train base images after foreground filtering
  - train images after augmentation multiplier
  - validation images
  - test images

Experiment attributes are centralized near the top of `main.py`:

```python
CONFIG = ExperimentConfig(
    dataset="dataset_drac",
    model_name="unet",
    backbone="efficientnet-b3",
    augment=True,
    aug_multiplier=10,
    epochs=100,
    batch_size=8,
    grad_accum_steps=1,
    lr=1e-4,
    best_metric="dice",
    use_amp=True,
    use_scheduler=True,
    scheduler_factor=0.5,
    scheduler_patience=3,
    min_lr=1e-7,
    required_conda_env="",
)
```

You can edit `CONFIG` directly for notebook-style experimentation, or override values through CLI arguments.

When `experiment_name=None`, `main.py` auto-generates a name from the resolved config:

```text
<dataset>_raw_<augment>_augx<aug_multiplier>_folds<k_folds>_ep<epochs>_bs<batch_size>_ga<grad_accum_steps>_lr<lr>_<amp>_<scheduler>_<model_token>
```

Example:

```text
tu_raw_with_augment_augx10_folds5_ep100_bs8_ga1_lr0p0001_amp_plateau_unet_efficientnetb3
```

Run example:

```bash
python main.py --dataset dataset_tu --augment --aug-multiplier 10
```

Recommended run command:

```bash
bash run_main.sh --config configs/default.toml
```

`run_main.sh` forwards all arguments to `main.py`. If `PIPELINE_CONDA_ENV` is set, it activates that conda environment first. `main.py` reads `[train]` from the TOML config first, then applies CLI overrides.

Disable augmentation:

```bash
python main.py --dataset dataset_drac --no-augment
```

Use a custom experiment folder name:

```bash
python main.py --experiment-name tu_trial_01
```

Model selection:

- `model_name` selects the architecture: `unet`, `unetplusplus`, `deeplabv3plus`, or `deeplabv3`.
- `backbone` selects the encoder: `efficientnet-b3`, `inceptionv4`, or `densenet169`.
- DeepLab architectures require a dilation-capable encoder in `segmentation_models_pytorch`; use `backbone="efficientnet-b3"` for DeepLab. Use `unet` or `unetplusplus` with `inceptionv4`/`densenet169`.
- `grad_accum_steps` keeps a larger effective training batch while the CUDA mini-batch stays small.

The script expects batches from the dataloader as dictionaries with `image` and `mask` tensors. Validation metrics use hole-filled postprocessed predictions. Training Dice uses thresholded predictions without hole filling to keep per-batch training feedback lightweight.

Training performance behavior:

- Uses metadata-free dataloader batches (`image` and `mask` only).
- Uses pinned-memory dataloading automatically when running on CUDA.
- Uses non-blocking device transfers for batch tensors.
- Uses AMP by default on CUDA to reduce GPU memory and improve throughput.
- Supports gradient accumulation with `grad_accum_steps`; the effective training batch is `batch_size * grad_accum_steps`.
- Uses `dice_score_from_logits` for train Dice instead of computing the full metric dictionary each batch.
- Keeps validation postprocessing because the selected validation metric drives model selection and scheduler behavior.

AMP can be disabled with:

```bash
python main.py --no-amp
```

Scheduler behavior:

- Enabled by default with `ReduceLROnPlateau`.
- Monitors the selected `best_metric`; `loss` uses `mode="min"` and all other metrics use `mode="max"`.
- Uses `scheduler_factor`, `scheduler_patience`, and `min_lr` from `CONFIG`.
- Can be disabled with `use_scheduler=False` or `python main.py --no-scheduler`.
- The learning rate is logged in `history_fold_<fold>.json` and plotted in `loss_dice_fold_<fold>.png`.

Environment behavior:

- `main.py` does not require a specific conda environment by default.
- The active conda environment, Python executable, and conda prefix are saved in `config.json`.
- To enforce a specific environment, set `required_conda_env="env_name"` in the TOML config or run with `--required-conda-env env_name`.

## `visualize_nearby.py`

Standalone evaluation and visualization script. It does not import or run `main.py`. The root-level `visualize.py` remains only as a compatibility wrapper around this script.

It uses:

- `utils.dataloader.create_dataloaders`
- `utils.metrics.compute_binary_mask_metrics`
- `utils.metrics.get_postprocessed_mask`
- `utils.model_factory.get_model`

Default behavior:

- reads fold weights from `model/<experiment_name>/weights/<fold>.pth`
- rebuilds the test dataloader for each fold
- runs each 1024x1024 test image through the model directly
- applies sigmoid, thresholding, and hole filling to predictions
- reuses that postprocessed mask for both metrics and saved images unless nearby representative-point filtering is enabled
- optionally keeps predicted connected components that are within a configured pixel radius of a representative localization point
- moves completed 1024x1024 predictions to CPU before metric and visualization work
- saves normal full-image visualization panels to `model/<experiment_name>/outputs/test_fold_<fold>/`
- saves nearby filtered panels to `model/<experiment_name>/outputs_nearby/test_fold_<fold>/`
- saves per-image metrics to `per_image_metrics.csv`
- saves normal fold-level metrics to `model/<experiment_name>/outputs/test_metrics.csv` and `test_metrics.json`
- saves nearby filtered fold-level metrics to `model/<experiment_name>/outputs_nearby/test_metrics.csv` and `test_metrics.json`
- saves average/std performance across folds to `test_metrics_summary.csv` and `test_metrics_summary.json` in the active output directory
- saves visualization settings to `visualize_config.json`

Run example:

```bash
python visualize_nearby.py --experiment-name tu_trial_01
```

Recommended run command:

```bash
bash run_visualize.sh --experiment-name tu_trial_01
```

`visualize_nearby.py` reads `[visualize]` from the TOML config first, then applies CLI overrides.

If `--experiment-name` is omitted, `visualize_nearby.py` generates the same default-style experiment name from its config. If `model/<experiment_name>/config.json` exists, dataset, model name, base directory, folds, batch size, and threshold are loaded from that saved config.

Representative-point filtering:

```toml
[visualize]
apply_nearby_filter = true
points_csv = "my_representative_points.csv"
points_image_name_template = "nv ({sample_id}).jpg"
radius_tolerance = 25.0
```

The points CSV is local-only and must contain `image_number`, `x`, and `y` columns. CSV coordinates use `x` as column and `y` as row in the 1024x1024 image. `points_image_name_template` controls how a dataset `sample_id` maps to the CSV `image_number` value. `radius_tolerance` controls how close a predicted component must be to at least one point.

Output separation:

- `apply_nearby_filter = false` writes to `model/<experiment_name>/outputs/`.
- `apply_nearby_filter = true` writes to `model/<experiment_name>/outputs_nearby/`.
- Running both modes on the same experiment does not overwrite the other mode's metrics or images.

Average performance across folds:

- Every visualization run writes `test_metrics_summary.csv`.
- The summary contains mean and standard deviation over all evaluated folds.
- The console prints average Dice and IoU after fold evaluation.

Average performance across datasets:

If the same setting is trained/evaluated for both datasets, the experiment folders should share the same suffix:

```text
model/tu_<setting_suffix>/
model/drac_<setting_suffix>/
```

Then run:

```bash
python visualize_nearby.py --summary-only --summarize-all-datasets --setting-suffix <setting_suffix>
```

If the current experiment name already starts with `tu_` or `drac_`, `--setting-suffix` can be omitted:

```bash
python visualize_nearby.py --experiment-name tu_<setting_suffix> --summary-only --summarize-all-datasets
```

The combined dataset summary is saved under:

```text
model/dataset_summaries/<setting_suffix>.csv
model/dataset_summaries/<setting_suffix>.json
```

For nearby filtered dataset summaries, filenames use the `_nearby` suffix:

```text
model/dataset_summaries/<setting_suffix>_nearby.csv
model/dataset_summaries/<setting_suffix>_nearby.json
```

For `tu` and `drac` rows, mean and standard deviation are computed over that dataset's folds. For the `all` row, mean and standard deviation are computed over all available fold rows from both datasets.

The script does not require a specific conda environment by default. To enforce one:

```bash
python visualize_nearby.py --experiment-name tu_trial_01 --required-conda-env rnv
```

## `specify_visualize.py`

Single-sample visualization script for checking one dataset image with one trained weight file. It does not run full fold evaluation and writes separate files under `model/<experiment_name>/extra_output/`.

Required values:

- `weight_path`: trained fold weight path, usually `model/<experiment_name>/weights/<fold>.pth`
- `dataset`: `tu`, `drac`, `dataset_tu`, or `dataset_drac`
- `sample_id`: image id inside the selected dataset

Run with the defaults from the script's parameter section:

```bash
python specify_visualize.py
```

Or pass the values explicitly:

```bash
python specify_visualize.py model/<experiment_name>/weights/1.pth tu 1
```

The detailed usage and output reference is in `SPECIFY_VISUALIZE.md`.
