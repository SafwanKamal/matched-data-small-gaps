#!/usr/bin/env bash
# Retrieval few-shot arm (workflow_retrieval_protocol.json). API only; saved replies are reused on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_retrieval "$@" >> logs/workflow-retrieval.log 2>&1
echo "WORKFLOW_RETRIEVAL_EXIT_$?" >> logs/workflow-retrieval.log
