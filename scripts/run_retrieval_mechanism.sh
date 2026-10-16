#!/usr/bin/env bash
# Usage: scripts/run_retrieval_mechanism.sh <stage>; logs to logs/d2-<stage>.log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.retrieval_mechanism "$@" > "logs/d2-$1${2:+-$2}.log" 2>&1
echo "D2_$1_EXIT_$?" >> "logs/d2-$1${2:+-$2}.log"
