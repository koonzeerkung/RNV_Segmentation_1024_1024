# specify_visualize.py

`specify_visualize.py` visualizes one selected sample with one trained weight file. It is useful when you want to inspect a single image without rerunning the full fold visualization pipeline.

## Required Configuration

The script is designed around three main values:

```python
DEFAULT_WEIGHT_PATH = "model/tu_0.4thresold_100epoch_efficientnet/weights/1.pth"
DEFAULT_DATASET = "tu"
DEFAULT_SAMPLE_ID = 1
```

- `DEFAULT_WEIGHT_PATH`: path to the trained weight file to load. In the normal pipeline this is `model/<experiment_name>/weights/<fold>.pth`.
- `DEFAULT_DATASET`: dataset to read from. Accepted values are `tu`, `drac`, `dataset_tu`, and `dataset_drac`.
- `DEFAULT_SAMPLE_ID`: image id inside the selected dataset, for example `1`.

You can either edit those defaults in the parameter section or pass the same values from the command line.

## Run

Using the parameter section defaults:

```bash
python specify_visualize.py
```

Using command-line values:

```bash
python specify_visualize.py model/<experiment_name>/weights/1.pth tu 1
```

If CUDA AMP causes a comparison mismatch or debugging noise, disable it:

```bash
python specify_visualize.py model/<experiment_name>/weights/1.pth tu 1 --no-amp
```

## Optional Parameters

- `--base-dir`: project/data root that contains `dataset_tu` and `dataset_drac`. Default is `.`.
- `--threshold`: sigmoid threshold used before hole-filling. Default is `0.4` unless changed in the parameter section.
- `--model-name`: model architecture: `unet`, `unetplusplus`, `deeplabv3plus`, or `deeplabv3`.
- `--backbone`: encoder backbone: `efficientnet-b3`, `inceptionv4`, or `densenet169`.
- `--amp` / `--no-amp`: enable or disable automatic mixed precision on CUDA.

The script also uses these constants from the parameter section:

- `OUTPUT_SUBDIR = "extra_output"`: output folder under the inferred experiment directory.
- `SAVE_PREDICTION_NPY = True`: also save the binary prediction mask as `.npy`.
- `FIGURE_DPI = 150`: PNG figure resolution.

## Outputs

The experiment directory is inferred from the weight path. For example:

```text
model/tu_0.4thresold_100epoch_efficientnet/weights/1.pth
```

writes to:

```text
model/tu_0.4thresold_100epoch_efficientnet/extra_output/
```

For `dataset_tu` sample `1`, outputs are:

```text
dataset_tu_sample_1.png
dataset_tu_sample_1_prediction.npy
dataset_tu_sample_1_metrics.csv
```

The PNG contains input, ground truth, prediction, error map, and confusion matrix when a ground-truth mask exists. If no ground truth is found, it saves only input and prediction.

## Metric Notes

`specify_visualize.py` uses the same full-image layout as the main pipeline: each `1024x1024` image is predicted in one forward pass, thresholded, and hole-filled.

If `extra_output/*_metrics.csv` differs slightly from `outputs/test_fold_*/per_image_metrics.csv`, first check whether `outputs/` is stale. The full visualizer may have been run earlier with older code or settings. Refresh normal outputs with:

```bash
bash run_visualize.sh --config configs/default.toml --experiment-name <experiment_name> --no-nearby-filter
```

Use `--no-nearby-filter` when comparing against `outputs/`. Nearby-filtered visualization writes to `outputs_nearby/`.

## Environment Notes

Run from the project root so dataset paths resolve correctly:

```bash
conda activate rnv
python specify_visualize.py
```

If `matplotlib` is missing in the active environment, install dependencies into the same interpreter that runs the script:

```bash
python -m pip install -r requirements.txt
```
