#!/usr/bin/env bash
# MultiDoGO real conversations (workflow_multidogo_protocol.json). Stage: hosted | train | report.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_multidogo "$1" >> logs/multidogo.log 2>&1
echo "MULTIDOGO_EXIT_$1_$?" >> logs/multidogo.log
