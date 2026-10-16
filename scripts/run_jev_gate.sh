#!/usr/bin/env bash
# Jev as scope gate (workflow_jev_gate_protocol.json). Jev API only; saved answers are reused on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_jev_gate "$@" >> logs/jev-gate.log 2>&1
echo "JEV_GATE_EXIT_$?" >> logs/jev-gate.log
