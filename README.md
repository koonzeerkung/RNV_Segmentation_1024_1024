# Performance Comparison Pipeline

This project trains and evaluates a U-Net retinal vessel segmentation model on `dataset_tu` or `dataset_drac`.

## Dataset Structure

Datasets are **not included in this repo** (see `.gitignore`). Place them yourself, either as
`dataset_tu` / `dataset_drac` at the repo root, or as `dataset/TU` / `dataset/DRAC`
(the loader tries both, case-insensitively). Each dataset has a `raw/` (image) and `GT/` (mask)
subfolder:

```text
dataset/
  TU/
    raw/          # 27 grayscale images, "nv (1).jpg" ... "nv (27).jpg"
    GT/            # 27 masks,          "gt (1).tif" ... "gt (27).tif"
  DRAC/
    raw/          # 42 grayscale images, arbitrary numeric filenames e.g. "082.png"
    GT/            # 42 masks with matching filenames,           e.g. "082.png"
```

- `dataset_tu`: 27 images, all 1024x1024, sample IDs 1-27 map directly to filenames
  (`nv (N).jpg` / `gt (N).tif`).
- `dataset_drac`: 42 images, all 1024x1024. Filenames are the *original* DRAC dataset IDs
  (not renamed to 1..42), so sample ID -> file is resolved by **numeric sort position**
  (1st file numerically = sample 1, etc.), not by matching the number in the filename.
- Fold assignment (fixed, see `utils/dataloader.py::get_kfold_chunks`):
  - `dataset_tu`: folds cover IDs 1-6, 7-12, 13-17, 18-22, 23-27
  - `dataset_drac`: folds cover IDs 1-9, 10-18, 19-26, 27-34, 35-42
  - For fold `n`: test = fold `n`, validation = fold `n+1` (wraps), train = remaining folds
- Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

## 1. Setup

Use any Python environment with the required packages. Conda example:

```bash
conda env create -f environment.yml
conda activate eeg
```

If you already have a working environment, just install the dependencies from `environment.yml` or your preferred package manager.

The pipeline does not require the environment to be named `eeg`.

For Google Colab, use [COLAB.md](COLAB.md). Short version:

```python
!pip install -r requirements.txt
!python main.py --config configs/colab.toml
```

Optional: if you want the helper scripts to activate a conda environment automatically, set:

```bash
export PIPELINE_CONDA_ENV=eeg
```

## 2. Edit Config

Open [configs/default.toml](configs/default.toml) and edit the `[train]` section:

```toml
[train]
dataset = "dataset_drac"
model_name = "unet"
backbone = "efficientnet-b3"
epochs = 100
batch_size = 8
lr = 0.0001
best_metric = "dice"
experiment_name = ""
```

Leave `experiment_name = ""` to auto-generate the experiment folder name from the config.

For a new experiment, copy the config first:

```bash
cp configs/default.toml configs/my_experiment.toml
```

Then edit `configs/my_experiment.toml`.

## 3. Train

```bash
./run_main.sh --config configs/my_experiment.toml
```

Outputs are saved under:

```text
model/<experiment_name>/
  config.json
  weights/
  outputs/
```

## 4. Visualize And Evaluate

Use the same config and the experiment name from training:

```bash
./run_visualize.sh --config configs/my_experiment.toml --experiment-name <experiment_name>
```

If you set a fixed `experiment_name` in the config, you can omit the CLI experiment name:

```bash
./run_visualize.sh --config configs/my_experiment.toml
```

Visualization writes fold-level metrics plus an average/std summary across folds:

```text
model/<experiment_name>/outputs/test_metrics.csv
model/<experiment_name>/outputs/test_metrics_summary.csv
model/<experiment_name>/outputs/test_metrics_summary.json
```

When nearby representative-point filtering is enabled, results are written separately so normal outputs are preserved:

```text
model/<experiment_name>/outputs_nearby/test_metrics.csv
model/<experiment_name>/outputs_nearby/test_metrics_summary.csv
model/<experiment_name>/outputs_nearby/test_metrics_summary.json
```

If you trained the same setting on both `tu` and `drac`, summarize both datasets with:

```bash
./run_visualize.sh --config configs/my_experiment.toml --summary-only --summarize-all-datasets --setting-suffix <setting_suffix>
```

The `all` row in the dataset summary reports mean +/- standard deviation across all available fold rows from both datasets.

Optional representative-point filtering can be enabled in `[visualize]` or from the CLI:

```toml
apply_nearby_filter = true
points_csv = "combined_representative_points.csv"
points_image_name_template = "nv ({sample_id}).jpg"
radius_tolerance = 25.0
```

When enabled, visualization keeps predicted connected components that are within `radius_tolerance` pixels of a representative point from the CSV. The CSV must contain `image_number`, `x`, and `y` columns.

For one specific image and one specific weight file, use `specify_visualize.py`. See [SPECIFY_VISUALIZE.md](SPECIFY_VISUALIZE.md) for the required parameters and output files.

## 5. Generate RNV Probability Heatmaps

`generate_rnv_heatmaps.py` uses the two included EfficientNet-B3 U-Net models:
`BEST_tu_unet_effib3_t0.35` and `BEST_drac_unet_effib3_t0.35`. A dataset input
automatically uses its matching model (TU for TU, DRAC for DRAC) and averages
that model's five cross-validation folds.

For any 1024x1024 retinal image:

```bash
conda activate eeg
python generate_rnv_heatmaps.py --image /path/to/image.png
```

Or for an image from a project dataset:

```bash
python generate_rnv_heatmaps.py --dataset tu --sample-id 1
python generate_rnv_heatmaps.py --dataset drac --sample-id 1
```

The default output is exactly one dark, color-scaled overlay PNG in
`rnv_heatmaps/` (use `--output-dir` to change it):

- `16_tu_rnv_probability_overlay.png` for `--dataset tu --sample-id 16`, for example.

The heatmap uses a fixed color scale of `P(RNV) = 0` to `1`, and the fundus
image underneath is deliberately darkened. Use `--models tu` or `--models drac`
when supplying an image directly, `--folds 1,3,5` for a subset of folds, or
`--save-probability-npy` to also save the float32 probability array. Every run
also saves `*_rnv_prediction_mask.png`: the model's binary prediction after
thresholding and hole filling.

To display only the model probability inside an externally supplied binary
region-of-interest mask, use `--roi-mask`. White pixels are kept and black
pixels are hidden; masks are resized with nearest-neighbor interpolation when
needed:

```bash
python generate_rnv_heatmaps.py --dataset tu --sample-id 27 --roi-mask /path/to/MASK.png
```

Add `--output-name my_heatmap` to choose the heatmap PNG filename.
Use `--background-brightness 0.55` to control the raw-image intensity beneath
the probability colors (`0` is black; `1` retains its original intensity).
Use `--background-gamma` below `1` to brighten low-intensity vessel detail.

To create one compact panel containing the raw image, probability heatmap, and
the binary mask filtered with every unclustered localizer point, run:

```bash
python generate_rnv_heatmaps.py --dataset tu --sample-id 27 --compact-points-filter
```

This uses the dataset's `*_RNV_Points_Unclustered.csv` file, a probability
threshold of `0.35`, and a 20-pixel component-to-point tolerance. Override
these with `--mask-threshold` and `--radius-tolerance` when needed.
It also writes `*_rnv_multiple_points_filtered_mask.png`: the final binary
model mask after the multiple-points connected-component filter.

## Useful Overrides

You can override config values from the command line:

```bash
./run_main.sh --config configs/my_experiment.toml --dataset dataset_tu --epochs 50 --batch-size 4
```

Choose which validation metric saves the best model:

```bash
./run_main.sh --config configs/my_experiment.toml --best-metric iou
```

Allowed values are `loss`, `dice`, `iou`, `precision`, `recall`, `specificity`, and `accuracy`. `loss` is minimized; the others are maximized.

Disable augmentation:

```bash
./run_main.sh --config configs/my_experiment.toml --no-augment
```

Disable AMP:

```bash
./run_main.sh --config configs/my_experiment.toml --no-amp
```
