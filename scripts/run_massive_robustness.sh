#!/usr/bin/env bash
# Usage: scripts/run_massive_robustness.sh <stage>; logs to logs/mr-<stage>.log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.massive_robustness "$@" > "logs/mr-$1${2:+-$2}.log" 2>&1
echo "MR_$1_EXIT_$?" >> "logs/mr-$1${2:+-$2}.log"
