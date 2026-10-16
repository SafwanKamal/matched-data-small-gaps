#!/usr/bin/env bash
# Workflow v4 (workflow_v4_protocol.json). API only, no GPU; saved responses are reused, so it can be rerun after a stop.
# Run from Git Bash:  bash scripts/run_workflow_v4.sh [all|a|b|c]
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_v4 "${1:-all}" >> logs/workflow-v4.log 2>&1
echo "WORKFLOW_V4_EXIT_$?" >> logs/workflow-v4.log
