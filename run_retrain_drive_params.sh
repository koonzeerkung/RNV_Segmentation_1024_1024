#!/usr/bin/env bash
#
# Full retrain of the six paper experiments at the Google Drive parameters, then evaluation.
#
# Reproduces the Drive runs exactly -- same seed (69), augmentation (x10), optimizer
# (Adam 1e-4, batch 8 x 4 accum), scheduler (ReduceLROnPlateau 0.5/5/1e-7), early stopping
# (25) and epoch ceilings (U-Net 100, U-Net++ and DeepLabv3+ 200), while fixing what was
# wrong with them.
#
# One deliberate departure from Drive: DeepLabv3+ trains and evaluates at threshold 0.45
# rather than 0.5. U-Net and U-Net++ keep Drive's 0.35. The validation sweep over the
# previous weights put DeepLabv3+'s optimum at 0.50 on TU and 0.40 on DRAC, so 0.45 sits
# between the two and no longer inherits a value that was read off test-split renders.
#
# What is fixed relative to Drive:
#
#   1. raw ground truth throughout. Requires GT_FILL_HOLES = False in utils/dataloader.py;
#      the script refuses to start otherwise, because training against filled masks is the
#      one thing that cannot be undone afterwards by re-running inference.
#   2. all five folds every time. The Drive tu_unet and drac_unet config.json carry a stale
#      "only_fold": 5 from fold-specific reruns; these configs have no such key.
#   3. both evaluation passes from one checkpoint, run immediately after that experiment's
#      training finishes. This is what makes the filter/no-filter delta a real measurement
#      -- the Drive DRAC DeepLabV3+ pair came from two different model states.
#
# Known residual, kept deliberately: main.py selects the best checkpoint by validation Dice
# computed at the config threshold, so weights stay mildly tuned toward 0.35 / 0.5.
# Removing it would need a neutral training threshold, which would no longer reproduce Drive.
#
# Results go to model/RETRAIN_* -- the existing model/BEST_* tree is never touched.
#
# Ordered cheapest first, so usable results arrive early. Measured from the previous run's
# history files, total is 6-8 days on an idle GPU.
#
# Usage:
#   nohup ./run_retrain_drive_params.sh > /dev/null 2>&1 &
#   SKIP_TRAINED=1 ./run_retrain_drive_params.sh   # resume: skip experiments already done

set -uo pipefail   # deliberately no -e: one failing experiment must not kill the queue

PIPELINE_CONDA_ENV="${PIPELINE_CONDA_ENV:-rnv}"
PYTHON="${PYTHON:-/home/wowza/miniconda3/envs/${PIPELINE_CONDA_ENV}/bin/python}"
SKIP_TRAINED="${SKIP_TRAINED:-0}"
POINTS_MODE="${POINTS_MODE:-multiple}"   # multiple = all localizer points; single = one per lesion

# main.py and visualize_nearby.py check CONDA_DEFAULT_ENV / CONDA_PREFIX, not sys.executable.
export CONDA_DEFAULT_ENV="${PIPELINE_CONDA_ENV}"
export CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"

cd "$(dirname "$0")" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="model/retrain_drive_logs_${STAMP}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${RUN_LOG}"; }

# config toml : experiment dir name, cheapest first
EXPERIMENTS=(
  "configs/retrain_tu_deeplabv3plus_effib3.toml:RETRAIN_tu_deeplabv3plus_efficientnetb3"
  "configs/retrain_tu_unet_effib3_t0.35.toml:RETRAIN_tu_unet_effib3_t0.35"
  "configs/retrain_drac_deeplabv3plus_effib3.toml:RETRAIN_drac_deeplabv3plus_efficientnetb3"
  "configs/retrain_tu_unetplusplus_effib3_t0.35.toml:RETRAIN_tu_unetplusplus_effib3_t0.35"
  "configs/retrain_drac_unet_effib3_t0.35.toml:RETRAIN_drac_unet_effib3_t0.35"
  "configs/retrain_drac_unetplusplus_effib3_t0.35.toml:RETRAIN_drac_unetplusplus_effib3_t0.35"
)

if [[ ! -x "${PYTHON}" ]]; then
  echo "python not found at ${PYTHON} -- set PYTHON=/path/to/python" >&2
  exit 1
fi

# Hard gate: training against hole-filled masks cannot be repaired by re-running inference.
GT_LINE="$(grep -m1 '^GT_FILL_HOLES' utils/dataloader.py)"
if [[ "${GT_LINE}" != "GT_FILL_HOLES = False" ]]; then
  echo "REFUSING TO START: utils/dataloader.py has '${GT_LINE}'." >&2
  echo "Set GT_FILL_HOLES = False before retraining, or the weights will be trained" >&2
  echo "against modified ground truth and the whole run is wasted." >&2
  exit 1
fi

log "python       : ${PYTHON}"
log "gt fill      : ${GT_LINE}"
log "experiments  : ${#EXPERIMENTS[@]}"
log "logs         : ${LOG_DIR}"
log "skip trained : ${SKIP_TRAINED}"
log "points mode  : ${POINTS_MODE}"
log ""

FAILED=()
START_ALL=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
  CONFIG="${entry%%:*}"
  EXP="${entry##*:}"
  EXP_DIR="model/${EXP}"

  log "=============================================================="
  log "${EXP}"

  if [[ ! -f "${CONFIG}" ]]; then
    log "  SKIP: ${CONFIG} missing"
    FAILED+=("${EXP} (no config)")
    continue
  fi

  # ---- train -------------------------------------------------------------
  if [[ "${SKIP_TRAINED}" == "1" && -f "${EXP_DIR}/weights/5.pth" ]]; then
    log "  [1/4] training already complete, skipping"
  else
    log "  [1/4] training"
    START=$(date +%s)
    "${PYTHON}" main.py --config "${CONFIG}" > "${LOG_DIR}/${EXP}_train.log" 2>&1
    RC=$?
    ELAPSED=$(( ($(date +%s) - START) / 60 ))
    if [[ ${RC} -ne 0 ]]; then
      log "  FAILED training (rc=${RC}, ${ELAPSED} min) -- see ${LOG_DIR}/${EXP}_train.log"
      FAILED+=("${EXP} (training)")
      continue
    fi
    log "  trained in ${ELAPSED} min"
  fi

  # ---- evaluate: both passes, same weights, config threshold -------------
  log "  [2/4] evaluation, no filter"
  "${PYTHON}" visualize_nearby.py --config "${CONFIG}" --experiment-name "${EXP}" \
    --no-nearby-filter > "${LOG_DIR}/${EXP}_nofilter.log" 2>&1
  if [[ $? -ne 0 ]]; then
    log "  FAILED no-filter pass -- see ${LOG_DIR}/${EXP}_nofilter.log"
    FAILED+=("${EXP} (no-filter pass)")
    continue
  fi

  log "  [3/4] evaluation, candidate-point filter"
  "${PYTHON}" visualize_nearby.py --config "${CONFIG}" --experiment-name "${EXP}" \
    --nearby-filter --points-mode "${POINTS_MODE}" > "${LOG_DIR}/${EXP}_filter.log" 2>&1
  if [[ $? -ne 0 ]]; then
    log "  FAILED filter pass -- see ${LOG_DIR}/${EXP}_filter.log"
    FAILED+=("${EXP} (filter pass)")
    continue
  fi

  # ---- validation threshold sweep, for the record ------------------------
  log "  [4/4] validation threshold sweep"
  "${PYTHON}" select_threshold_val.py --experiment "${EXP}" \
    > "${LOG_DIR}/${EXP}_sweep.log" 2>&1
  if [[ $? -ne 0 ]]; then
    log "  WARNING: sweep failed -- see ${LOG_DIR}/${EXP}_sweep.log (results still usable)"
  fi

  log "  done"
done

TOTAL=$(( ($(date +%s) - START_ALL) / 60 ))
log ""
log "=============================================================="
log "elapsed: ${TOTAL} min"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "FAILURES (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do log "  - ${f}"; done
else
  log "all experiments completed"
fi

log ""
log "building summary"
"${PYTHON}" summarize_threshold_eval.py \
  --prefix RETRAIN_ \
  --threshold "0.35 unet/unet++, 0.45 deeplabv3+" \
  --points-mode "${POINTS_MODE}" \
  --output "${LOG_DIR}/summary_retrain.csv" \
  2>&1 | tee -a "${RUN_LOG}"

log ""
log "summary : ${LOG_DIR}/summary_retrain.csv"
log "logs    : ${LOG_DIR}"
