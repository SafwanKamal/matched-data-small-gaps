#!/usr/bin/env bash
# Usage: scripts/run_option_count.sh <stage> [model]; logs to logs/e-<stage>[-model].log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.option_count "$@" > "logs/e-$1${2:+-$2}.log" 2>&1
echo "E_$1_EXIT_$?" >> "logs/e-$1${2:+-$2}.log"
