# RNV Segmentation 1024x1024 Pipeline

This project trains and evaluates retinal vessel segmentation models on full 1024x1024 images. It supports `unet`, `unetplusplus`, `deeplabv3plus`, and `deeplabv3` with configurable encoder backbones.

Datasets and trained model outputs are intentionally not included in this repository. Place them locally after cloning.

## 1. Setup

Clone the repository and create the conda environment:

```bash
git clone https://github.com/koonzeerkung/RNV_Segmentation_1024_1024.git
cd RNV_Segmentation_1024_1024
conda env create -f environment.yml
conda activate rnv
```

Or install dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

The helper scripts can auto-activate a conda environment when `PIPELINE_CONDA_ENV` is set:

```bash
export PIPELINE_CONDA_ENV=rnv
```

For Google Colab, use [COLAB.md](COLAB.md). Short version:

```python
!pip install -r requirements.txt
!python main.py --config configs/colab.toml
```

## 2. Add Datasets

Put datasets under one of the supported local layouts.

Current 1024x1024 layout:

```text
dataset/
  TU/
    raw/
      nv (1).jpg
      nv (2).jpg
      ...
    GT/
      gt (1).tif
      gt (2).tif
      ...
  DRAC/
    raw/
      1.png
      2.png
      ...
    GT/
      1.png
      2.png
      ...
```

Legacy layout is also supported:

```text
dataset_tu/
  raw/
    nv (1).jpg
    nv (2).jpg
    ...
  gt/
    gt (1).tif
    gt (2).tif
    ...
dataset_drac/
  raw/
    1.png
    2.png
    ...
  gt/
    1.png
    2.png
    ...
```

Accepted dataset names in config or CLI are `tu`, `drac`, `dataset_tu`, and `dataset_drac`.

The loader expects each raw image and ground-truth mask to share the same numeric sample id. Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`. Images and masks must be 1024x1024 pixels.

The built-in five-fold split expects:

- TU: sample ids `1` through `27`
- DRAC: sample ids `1` through `42`

Datasets are local-only and are ignored by git.

## 3. Edit Config

Open [configs/default.toml](configs/default.toml) and edit the `[train]` section:

```toml
[train]
dataset = "tu"
model_name = "unet"
backbone = "efficientnet-b3"
epochs = 200
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

## 4. Train

```bash
bash run_main.sh --config configs/my_experiment.toml
```

Outputs are saved under:

```text
model/<experiment_name>/
  config.json
  weights/
  outputs/
```

Each fold summary records when the best model was saved:

```text
model/<experiment_name>/outputs/best_model_timing_fold_<fold>.json
model/<experiment_name>/outputs/kfold_summary.json
```

## 5. Visualize And Evaluate

Use the same config and the experiment name from training:

```bash
bash run_visualize.sh --config configs/my_experiment.toml --experiment-name <experiment_name>
```

If you set a fixed `experiment_name` in the config, you can omit the CLI experiment name:

```bash
bash run_visualize.sh --config configs/my_experiment.toml
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
bash run_visualize.sh --config configs/my_experiment.toml --summary-only --summarize-all-datasets --setting-suffix <setting_suffix>
```

The `all` row in the dataset summary reports mean +/- standard deviation across all available fold rows from both datasets.

Optional representative-point filtering can be enabled in `[visualize]` or from the CLI:

```toml
apply_nearby_filter = true
points_csv = "my_representative_points.csv"
points_image_name_template = "nv ({sample_id}).jpg"
radius_tolerance = 25.0
```

When enabled, visualization keeps predicted connected components that are within `radius_tolerance` pixels of a representative point from the CSV. The CSV is local-only, is not included in this repository, and must contain `image_number`, `x`, and `y` columns.

For one specific image and one specific weight file, use `specify_visualize.py`. See [SPECIFY_VISUALIZE.md](SPECIFY_VISUALIZE.md) for the required parameters and output files.

## 6. Continuous Experiment Queue

The Python queue runner trains one experiment, visualizes it, then moves to the next experiment:

```bash
python run_experiment_queue.py --dry-run
python run_experiment_queue.py
```

Default order is TU first, then DRAC. Within each dataset it starts with DeepLab, then U-Net++, then U-Net. DeepLab currently runs only with `efficientnet-b3`; unsupported DeepLab/backbone combinations are skipped with a clear message.

See [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md) for details.

## Useful Overrides

You can override config values from the command line:

```bash
bash run_main.sh --config configs/my_experiment.toml --dataset dataset_tu --epochs 50 --batch-size 4
```

Choose which validation metric saves the best model:

```bash
bash run_main.sh --config configs/my_experiment.toml --best-metric iou
```

Allowed values are `loss`, `dice`, `iou`, `precision`, `recall`, `specificity`, and `accuracy`. `loss` is minimized; the others are maximized.

Disable augmentation:

```bash
bash run_main.sh --config configs/my_experiment.toml --no-augment
```

Disable AMP:

```bash
bash run_main.sh --config configs/my_experiment.toml --no-amp
```
