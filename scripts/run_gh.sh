#!/usr/bin/env bash
# Usage: scripts/run_gh_runs.sh <stage>; logs to logs/gh-<stage>.log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.gh_runs "$@" > "logs/gh-$1${2:+-$2}.log" 2>&1
echo "GH_$1_EXIT_$?" >> "logs/gh-$1${2:+-$2}.log"
