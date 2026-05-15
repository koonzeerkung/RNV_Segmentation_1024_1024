# Performance Comparison Pipeline

This project trains and evaluates a U-Net segmentation pipeline on `dataset_tu` or `dataset_drac`.

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
epochs = 100
batch_size = 8
lr = 0.0001
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

## Useful Overrides

You can override config values from the command line:

```bash
./run_main.sh --config configs/my_experiment.toml --dataset dataset_tu --epochs 50 --batch-size 4
```

Disable augmentation:

```bash
./run_main.sh --config configs/my_experiment.toml --no-augment
```

Disable AMP:

```bash
./run_main.sh --config configs/my_experiment.toml --no-amp
```
