# Kaggle Guide

This guide runs the pipeline in a Kaggle notebook with Kaggle's free GPU accelerator.

## 1. Prepare The Data

Create a Kaggle Dataset that contains one of these supported layouts:

```text
dataset_tu/
  raw/
  gt/
dataset_drac/
  raw/
  gt/
```

or:

```text
dataset/
  TU/
    raw/
    GT/
  DRAC/
    raw/
    GT/
```

After attaching the dataset to your notebook, Kaggle mounts it under:

```text
/kaggle/input/<dataset-slug>
```

For example, if the slug is `rnv-segmentation-data`, the config should use:

```toml
base_dir = "/kaggle/input/rnv-segmentation-data"
```

Kaggle input folders are read-only. Keep outputs under `/kaggle/working`.

## 2. Create A Notebook

1. Open Kaggle.
2. Create a new notebook.
3. In the right sidebar, choose an accelerator such as `GPU T4 x2` or `GPU P100` if available.
4. Attach your dataset under `Input`.
5. Enable internet only if you want Kaggle to clone the repo and install packages directly.

## 3. Get The Code Into Kaggle

With internet enabled:

```python
!git clone https://github.com/koonzeerkung/RNV_Segmentation_1024_1024.git
%cd RNV_Segmentation_1024_1024
```

If internet is disabled, upload the repository as a Kaggle Dataset and copy it into `/kaggle/working`:

```python
!cp -r /kaggle/input/<repo-dataset-slug>/RNV_Segmentation_1024_1024 /kaggle/working/
%cd /kaggle/working/RNV_Segmentation_1024_1024
```

## 4. Install Dependencies

Kaggle usually already includes PyTorch. Install only the missing project packages first:

```python
!pip install segmentation-models-pytorch timm albumentations opencv-python-headless
```

If you prefer to use the full requirements file:

```python
!pip install -r requirements.txt
```

For an offline notebook, add a wheelhouse as a Kaggle Dataset and install from it:

```python
!pip install --no-index --find-links /kaggle/input/<wheelhouse-slug>/wheels segmentation-models-pytorch timm albumentations opencv-python-headless
```

## 5. Edit The Kaggle Config

Copy the example config:

```python
!cp configs/kaggle.toml configs/my_kaggle_experiment.toml
```

Open `configs/my_kaggle_experiment.toml` in the Kaggle file browser and edit:

```toml
dataset = "dataset_tu"
base_dir = "/kaggle/input/<your-dataset-slug>"
model_root = "/kaggle/working/model"
encoder_weights = ""
batch_size = 1
grad_accum_steps = 8
```

Use `dataset = "dataset_drac"` for DRAC.

`encoder_weights = ""` prevents `segmentation-models-pytorch` from downloading ImageNet encoder weights. If you enable Kaggle internet and want pretrained encoders, use:

```toml
encoder_weights = "imagenet"
```

## 6. Run A Smoke Test

Before starting a long run, check that imports, data paths, model creation, and output writes work:

```python
!python main.py --config configs/my_kaggle_experiment.toml --epochs 1 --k-folds 1 --experiment-name kaggle_smoke_test
```

If you get CUDA out-of-memory, lower `batch_size` to `1`, keep `use_amp = true`, and increase `grad_accum_steps` only if you need a larger effective batch.

## 7. Train

Run the full experiment:

```python
!python main.py --config configs/my_kaggle_experiment.toml --experiment-name kaggle_tu_unet
```

Outputs are written to:

```text
/kaggle/working/model/kaggle_tu_unet/
  config.json
  weights/
  outputs/
```

## 8. Visualize And Evaluate

Use the same config and experiment name:

```python
!python visualize_nearby.py --config configs/my_kaggle_experiment.toml --experiment-name kaggle_tu_unet
```

The evaluation outputs are under:

```text
/kaggle/working/model/kaggle_tu_unet/outputs/
```

## 9. Save Results

Kaggle preserves `/kaggle/working` as notebook output after the run finishes. To make a single archive:

```python
!tar -czf /kaggle/working/rnv_results.tar.gz -C /kaggle/working model
```

Download `rnv_results.tar.gz` from the notebook output panel.

## Troubleshooting

- `Missing raw directory`: update `base_dir` or the dataset layout.
- `No image file found`: filenames must contain the numeric sample id expected by the built-in folds.
- `CUDA out of memory`: reduce `batch_size`, keep AMP enabled, or use a smaller backbone.
- Download errors while creating the model: keep `encoder_weights = ""` or enable Kaggle internet.
- Package install errors with internet off: attach wheels as a Kaggle Dataset and install with `--no-index --find-links`.
