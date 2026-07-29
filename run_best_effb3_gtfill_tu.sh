#!/usr/bin/env bash
# Stage 2 of the hole-filled-GT reruns: the three TU EfficientNet-B3 BEST_* experiments.
# Same contract as run_best_effb3_gtfill.sh (which handles DRAC): per experiment,
# train -> visualize (normal) -> visualize (cluster/nearby filter), overwriting
# model/BEST_tu_* in place, using each experiment's own config for all three stages.
#
# TU's measured GT-fill effect is ~0 (-0.0003..+0.0008 dice, see gt_fillhole_check/rescore),
# so this stage exists for protocol consistency: every reported number then comes from
# models trained and scored on hole-filled GT.
#
# Resumable: each finished stage drops a stamp in the log dir and is skipped on rerun.
# Force a redo with:  rm model/best_effb3_gtfill_logs/<stage>.done

set -uo pipefail

LOG_DIR="model/best_effb3_gtfill_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/best_tu_deeplabv3plus_effib3.toml"          # fastest of the three
  "configs/best_tu_unet_effib3_t0.35.toml"
  "configs/best_tu_unetplusplus_effib3_t0.35.toml"
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

  run_stage "${name}_viz" ./run_visualize.sh --config "$config" --no-nearby-filter
  run_stage "${name}_viz_nearby" ./run_visualize.sh --config "$config" --nearby-filter
done

echo "==================== TU queue finished $(date -Is) ===================="
