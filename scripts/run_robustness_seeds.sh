#!/usr/bin/env bash
# robustness_protocol.json, part run by bakeoff.robustness_seeds. Finished work is skipped on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.robustness_seeds >> logs/robustness-seeds.log 2>&1
echo "ROBUSTNESS_SEEDS_EXIT_$?" >> logs/robustness-seeds.log
