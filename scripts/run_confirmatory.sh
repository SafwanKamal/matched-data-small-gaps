#!/usr/bin/env bash
# Usage: scripts/run_confirmatory.sh <stage> [args]; logs to logs/confirmatory-<stage>.log
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
log="logs/confirmatory-$1${2:+-$2}.log"
.venv/Scripts/python.exe -m bakeoff.confirmatory "$@" > "$log" 2>&1
echo "CONFIRMATORY_$1_EXIT_$?" >> "$log"
