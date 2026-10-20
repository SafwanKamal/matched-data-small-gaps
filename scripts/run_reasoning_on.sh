#!/usr/bin/env bash
# Usage: scripts/run_reasoning_on.sh <stage> [model]; logs to logs/r-<stage>[-model].log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.reasoning_on "$@" > "logs/r-$1${2:+-$2}.log" 2>&1
echo "R_$1_EXIT_$?" >> "logs/r-$1${2:+-$2}.log"
