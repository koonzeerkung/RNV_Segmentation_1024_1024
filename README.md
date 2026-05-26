# Performance Comparison Pipeline

This project trains and evaluates a U-Net retinal vessel segmentation model on `dataset_tu` or `dataset_drac`.

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
