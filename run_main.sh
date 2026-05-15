#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PIPELINE_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PIPELINE_CONDA_ENV}"
fi

python main.py "$@"
