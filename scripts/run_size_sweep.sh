#!/usr/bin/env bash
# Ettin size sweep (size_sweep_protocol.json). One GPU job; finished jobs are skipped, so it can be rerun after a stop.
# Run from Git Bash:  bash scripts/run_size_sweep.sh [all|encoders|decoders]
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.size_sweep "${1:-all}" >> logs/size-sweep.log 2>&1
echo "SIZE_SWEEP_EXIT_$?" >> logs/size-sweep.log
