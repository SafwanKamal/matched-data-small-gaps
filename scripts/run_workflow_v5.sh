#!/usr/bin/env bash
# Workflow v5 (workflow_v5_protocol.json). API only. Waits for workflow v4 to finish first so the two do not
# both write the spend file. Saved replies are reused on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
until [ "$(grep -c WORKFLOW_V4_EXIT_ logs/workflow-v4.log)" -ge 3 ]; do sleep 30; done
.venv/Scripts/python.exe -m bakeoff.workflow_v5 "${1:-all}" >> logs/workflow-v5.log 2>&1
echo "WORKFLOW_V5_EXIT_$?" >> logs/workflow-v5.log
