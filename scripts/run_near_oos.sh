#!/usr/bin/env bash
# Near-domain out-of-scope on BANKING77-OOS (workflow_near_oos_protocol.json). Stage: hosted | train | report.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_near_oos "$1" >> logs/near-oos.log 2>&1
echo "NEAR_OOS_EXIT_$1_$?" >> logs/near-oos.log
