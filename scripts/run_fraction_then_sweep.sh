#!/usr/bin/env bash
# Train and score the data-fraction specialists (data_fraction_protocol.json), then resume the size sweep.
# One GPU job at a time; both steps skip finished work.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.data_fraction train >> logs/data-fraction.log 2>&1
echo "DATA_FRACTION_EXIT_$?" >> logs/data-fraction.log
bash scripts/run_size_sweep.sh all
