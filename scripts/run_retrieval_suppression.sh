#!/usr/bin/env bash
# Usage: scripts/run_retrieval_suppression.sh <stage>; logs to logs/rs-<stage>.log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.retrieval_suppression "$@" > "logs/rs-$1.log" 2>&1
echo "RS_$1_EXIT_$?" >> "logs/rs-$1.log"
