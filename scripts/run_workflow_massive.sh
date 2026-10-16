#!/usr/bin/env bash
# MASSIVE replication of the workflow ladder (workflow_massive_protocol.json). API only unless the stage is
# 'decoder', which scores the fine-tuned decoder on the GPU. Saved replies are reused on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_massive "${1:-all}" >> logs/workflow-massive.log 2>&1
echo "WORKFLOW_MASSIVE_EXIT_$?" >> logs/workflow-massive.log
