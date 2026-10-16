#!/usr/bin/env bash
# Context-handoff test (handoff_protocol.json). API only; saved replies are reused on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.handoff "$@" >> logs/handoff.log 2>&1
echo "HANDOFF_EXIT_$?" >> logs/handoff.log
