#!/usr/bin/env bash
#
# Overnight re-evaluation of the six paper experiments at a single validation-selected
# threshold (default 0.45), replacing the per-model 0.35 / 0.5 thresholds.
#
# For each experiment it runs TWO passes from the SAME weights:
#   1. no filter        -> model/<exp>/outputs/
#   2. candidate-point  -> model/<exp>/outputs_nearby_<points_mode>/
# Running both from one fixed checkpoint is what fixes the invalid DRAC DeepLabV3+ pair
# recorded in ref/PAPER_TODO.md (recall rose after filtering because the two passes came
# from different model states).
#
# Inference only. No training. Existing outputs/ and outputs_nearby*/ are backed up to
# outputs_backup_<threshold>_<timestamp>/ before anything is overwritten.
#
# Usage:
#   ./run_eval_t045.sh                 # threshold 0.45, all six experiments
#   THRESHOLD=0.50 ./run_eval_t045.sh  # different threshold
#   nohup ./run_eval_t045.sh > /dev/null 2>&1 &   # detach for overnight

set -uo pipefail   # deliberately no -e: one failing experiment must not kill the run

THRESHOLD="${THRESHOLD:-0.45}"
POINTS_MODE="${POINTS_MODE:-multiple}"   # multiple = all localizer points; single = one per lesion
PIPELINE_CONDA_ENV="${PIPELINE_CONDA_ENV:-rnv}"
PYTHON="${PYTHON:-/home/wowza/miniconda3/envs/${PIPELINE_CONDA_ENV}/bin/python}"

# visualize_nearby.py::ensure_conda_env() reads CONDA_DEFAULT_ENV / CONDA_PREFIX, not
# sys.executable, so calling the env's interpreter directly is not enough -- without these
# it aborts with "active environment is 'base'".
export CONDA_DEFAULT_ENV="${PIPELINE_CONDA_ENV}"
export CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"

cd "$(dirname "$0")" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="model/eval_t${THRESHOLD}_logs_${STAMP}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RUN_LOG}"; }

# config toml : experiment dir name
EXPERIMENTS=(
  "configs/best_tu_unet_effib3_t0.35.toml:BEST_tu_unet_effib3_t0.35"
  "configs/best_tu_unetplusplus_effib3_t0.35.toml:BEST_tu_unetplusplus_effib3_t0.35"
  "configs/best_tu_deeplabv3plus_effib3.toml:BEST_tu_deeplabv3plus_efficientnetb3"
  "configs/best_drac_unet_effib3_t0.35.toml:BEST_drac_unet_effib3_t0.35"
  "configs/best_drac_unetplusplus_effib3_t0.35.toml:BEST_drac_unetplusplus_effib3_t0.35"
  "configs/best_drac_deeplabv3plus_effib3.toml:BEST_drac_deeplabv3plus_efficientnetb3"
)

if [[ ! -x "${PYTHON}" ]]; then
  echo "python not found at ${PYTHON} -- set PYTHON=/path/to/python" >&2
  exit 1
fi

log "threshold      : ${THRESHOLD}"
log "points mode    : ${POINTS_MODE}"
log "python         : ${PYTHON}"
log "experiments    : ${#EXPERIMENTS[@]}"
log "logs           : ${LOG_DIR}"
log ""

FAILED=()

for entry in "${EXPERIMENTS[@]}"; do
  CONFIG="${entry%%:*}"
  EXP="${entry##*:}"
  EXP_DIR="model/${EXP}"

  log "=============================================================="
  log "${EXP}"

  if [[ ! -d "${EXP_DIR}/weights" ]]; then
    log "  SKIP: ${EXP_DIR}/weights missing"
    FAILED+=("${EXP} (no weights)")
    continue
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    log "  SKIP: ${CONFIG} missing"
    FAILED+=("${EXP} (no config)")
    continue
  fi

  # Preserve the current (threshold 0.35 / 0.5) results before overwriting them.
  BACKUP="${EXP_DIR}/outputs_backup_pre_t${THRESHOLD}_${STAMP}"
  for sub in outputs outputs_nearby "outputs_nearby_${POINTS_MODE}"; do
    if [[ -d "${EXP_DIR}/${sub}" ]]; then
      mkdir -p "${BACKUP}"
      cp -a "${EXP_DIR}/${sub}" "${BACKUP}/${sub}"
      log "  backed up ${sub} -> ${BACKUP}/${sub}"
    fi
  done

  # Pass 1: no candidate-point filter.
  log "  [1/2] no filter, threshold ${THRESHOLD}"
  "${PYTHON}" visualize_nearby.py \
    --config "${CONFIG}" \
    --experiment-name "${EXP}" \
    --threshold "${THRESHOLD}" \
    --no-nearby-filter \
    > "${LOG_DIR}/${EXP}_nofilter.log" 2>&1
  RC1=$?
  if [[ ${RC1} -ne 0 ]]; then
    log "  FAILED (rc=${RC1}) -- see ${LOG_DIR}/${EXP}_nofilter.log"
    FAILED+=("${EXP} (no-filter pass)")
    continue
  fi

  # Pass 2: candidate-point filter, same weights, same threshold.
  log "  [2/2] candidate-point filter, threshold ${THRESHOLD}"
  "${PYTHON}" visualize_nearby.py \
    --config "${CONFIG}" \
    --experiment-name "${EXP}" \
    --threshold "${THRESHOLD}" \
    --nearby-filter \
    --points-mode "${POINTS_MODE}" \
    > "${LOG_DIR}/${EXP}_filter.log" 2>&1
  RC2=$?
  if [[ ${RC2} -ne 0 ]]; then
    log "  FAILED (rc=${RC2}) -- see ${LOG_DIR}/${EXP}_filter.log"
    FAILED+=("${EXP} (filter pass)")
    continue
  fi

  log "  done"
done

log ""
log "=============================================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "FAILURES (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do log "  - ${f}"; done
else
  log "all experiments completed"
fi

log ""
log "building summary"
"${PYTHON}" summarize_threshold_eval.py \
  --threshold "${THRESHOLD}" \
  --points-mode "${POINTS_MODE}" \
  --output "${LOG_DIR}/summary_t${THRESHOLD}.csv" \
  2>&1 | tee -a "${RUN_LOG}"

log ""
log "summary : ${LOG_DIR}/summary_t${THRESHOLD}.csv"
log "logs    : ${LOG_DIR}"
