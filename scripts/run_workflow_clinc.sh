#!/usr/bin/env bash
# CLINC150 workflow (workflow_clinc_protocol.json), API stages. Saved replies are reused on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_clinc "${1:-all}" >> logs/workflow-clinc.log 2>&1
echo "WORKFLOW_CLINC_RERUN_EXIT_$?" >> logs/workflow-clinc.log
