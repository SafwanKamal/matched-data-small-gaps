#!/usr/bin/env bash
# Usage: scripts/run_jev_rerun.sh <stage>; logs to logs/jev-rerun-<stage>.log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.jev_rerun "$@" > "logs/jev-rerun-$1.log" 2>&1
echo "JEV_$1_EXIT_$?" >> "logs/jev-rerun-$1.log"
