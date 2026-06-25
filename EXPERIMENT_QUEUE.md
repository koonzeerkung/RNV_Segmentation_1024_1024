# Continuous Experiment Queue

Run the full queue with:

```bash
python run_experiment_queue.py
```

The default queue order is:

1. TU
2. DRAC

Within each dataset it runs:

1. `deeplabv3plus`
2. `unetplusplus`
3. `unet`

Each architecture is paired with:

1. `efficientnet-b3`
2. `inceptionv4`
3. `densenet169`

For each runnable experiment, the tool runs training first, then visualization, then moves to the next experiment.

Outputs:

- Experiment folders: `model/<experiment_name>/`
- Queue state: `model/experiment_queue_state.json`
- Logs: `model/experiment_queue_logs/`

Useful commands:

```bash
# Preview the queue without running training.
python run_experiment_queue.py --dry-run

# Force visualization without the nearby filter.
python run_experiment_queue.py --no-nearby-filter

# Pass extra training args through to main.py.
python run_experiment_queue.py --train-arg=--epochs --train-arg=50

# Pass extra visualization args through to visualize_nearby.py.
python run_experiment_queue.py --visualize-arg=--max-images-per-fold --visualize-arg=5

# Continue to the next experiment if one experiment fails.
python run_experiment_queue.py --continue-on-error
```

By default, completed experiments are skipped when their visualization metrics already exist. If training weights already exist but visualization metrics do not, the tool runs visualization only.

Note: the current model factory rejects DeepLab with `inceptionv4` and `densenet169`, because those encoders do not support the dilated mode required by DeepLab in `segmentation_models_pytorch`. The queue therefore requests 18 combinations by default but runs 14 valid combinations unless this model support changes.
