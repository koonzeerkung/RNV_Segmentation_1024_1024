# Google Colab Guide

Colab does not need conda. Use `pip`, mount Google Drive if you want outputs to persist, then run the normal scripts.

## 1. Clone Or Upload Project

In a Colab notebook cell:

```python
!git clone <YOUR_REPO_URL> performance_comparison
%cd performance_comparison
```

If you are not using GitHub, upload the project folder to Colab or Google Drive, then `%cd` into it.

## 2. Install Dependencies

```python
!pip install -r requirements.txt
```

If Colab already has a compatible PyTorch build, this may only install the missing packages.

## 3. Mount Google Drive

This is recommended because `/content` is temporary.

```python
from google.colab import drive
drive.mount("/content/drive")
```

The example config writes experiment outputs to:

```text
/content/drive/MyDrive/rnv_pipeline/model
```

## 4. Put Dataset In The Right Place

The pipeline expects:

```text
dataset_drac/
  raw/
  gt/
dataset_tu/
  raw/
  gt/
```

If the datasets are inside the cloned project, keep:

```toml
base_dir = "."
```

If the datasets are on Drive, set `base_dir` in `configs/colab.toml`, for example:

```toml
base_dir = "/content/drive/MyDrive/rnv_pipeline/data"
```

## 5. Edit Config

Use the Colab example config:

```python
!cp configs/colab.toml configs/my_colab_experiment.toml
```

Edit it in the Colab file browser, especially:

```toml
dataset = "dataset_drac"
model_name = "unet"
backbone = "efficientnet-b3"
base_dir = "."
model_root = "/content/drive/MyDrive/rnv_pipeline/model"
epochs = 100
batch_size = 8
experiment_name = ""
```

Leave `experiment_name = ""` to auto-generate the experiment folder name.

## 6. Train

```python
!python main.py --config configs/my_colab_experiment.toml
```

For a quick smoke test:

```python
!python main.py --config configs/my_colab_experiment.toml --epochs 1 --k-folds 1 --experiment-name colab_smoke_test
```

## 7. Visualize And Evaluate

If you used a fixed experiment name:

```python
!python visualize_nearby.py --config configs/my_colab_experiment.toml --experiment-name colab_smoke_test
```

If training auto-generated the name, check the printed experiment name or the folder under `model_root`, then pass it:

```python
!python visualize_nearby.py --config configs/my_colab_experiment.toml --experiment-name <experiment_name>
```

To enable nearby representative-point filtering, set `apply_nearby_filter = true`, `points_csv`, and `radius_tolerance` in `[visualize]`, or pass `--nearby-filter --points-csv <csv_path>`. Normal outputs stay under `outputs/`; filtered outputs are written to `outputs_nearby/`.

## Notes

- Colab runtime storage is temporary. Use Google Drive for `model_root`.
- The `required_conda_env` setting should stay empty on Colab.
- If Colab runs out of GPU memory, reduce `batch_size`.
- If data loading is unstable, set `num_workers = 0`.
