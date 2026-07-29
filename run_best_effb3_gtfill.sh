#!/usr/bin/env bash
# Rerun the DRAC EfficientNet-B3 BEST_* experiments with hole-filled ground truth
# (utils/dataloader.py::GT_FILL_HOLES), overwriting the existing model/BEST_drac_* dirs.
#
# TU is deliberately excluded: rescore_gt_fill_holes.py measured the GT-fill effect on TU
# at -0.0003..+0.0008 dice (its GT is already solid), versus +0.0102..+0.0119 dice and
# +0.019..+0.026 precision on DRAC. TU configs still exist under configs/best_tu_*.toml.
#
# Per experiment: train -> visualize (normal) -> visualize (cluster/nearby filter),
# all three using that experiment's own config, so thresholds stay per-experiment.
# Ordered by the measured train time of the previous identical runs, min -> max.
#
# Resumable: each finished stage drops a stamp in the log dir and is skipped on rerun.
# Force a redo with:  rm model/best_effb3_gtfill_logs/<stage>.done

set -uo pipefail

LOG_DIR="model/best_effb3_gtfill_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/best_drac_unet_effib3_t0.35.toml"           # ~26.2 h previously
  "configs/best_drac_deeplabv3plus_effib3.toml"        # ~27.4 h
  "configs/best_drac_unetplusplus_effib3_t0.35.toml"   # ~31.9 h
)

run_stage() {
  local stamp="$LOG_DIR/$1.done"
  local log="$LOG_DIR/$1.log"
  shift

  if [[ -f "$stamp" ]]; then
    echo "[skip] $(basename "$stamp" .done) already done"
    return 0
  fi

  echo "[start] $(basename "$log" .log) :: $* :: $(date -Is)"
  if "$@" >>"$log" 2>&1; then
    date -Is >"$stamp"
    echo "[ok]    $(basename "$log" .log) :: $(date -Is)"
    return 0
  fi

  echo "[FAIL]  $(basename "$log" .log) :: see $log"
  return 1
}

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .toml)"
  echo "==================== $name ===================="

  if ! run_stage "${name}_train" ./run_main.sh --config "$config"; then
    echo "training failed for $name — skipping its visualizations"
    continue
  fi

  # Normal visualization: model/<exp>/outputs
  run_stage "${name}_viz" ./run_visualize.sh --config "$config" --no-nearby-filter

  # Cluster (representative-point) filter: model/<exp>/outputs_nearby
  run_stage "${name}_viz_nearby" ./run_visualize.sh --config "$config" --nearby-filter
done

echo "==================== queue finished $(date -Is) ===================="
