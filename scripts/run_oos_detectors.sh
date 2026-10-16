#!/usr/bin/env bash
# Encoder-side out-of-scope detectors (oos_detectors_protocol.json). Processor only, no API call.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.oos_detectors "$@" >> logs/oos-detectors.log 2>&1
echo "OOS_DETECTORS_EXIT_$?" >> logs/oos-detectors.log
