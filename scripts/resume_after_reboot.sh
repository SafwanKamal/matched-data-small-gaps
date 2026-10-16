#!/usr/bin/env bash
# Restart the work that was paused for the reboot on 2026-10-11: workflow v3 (the routing test).
# Run from Git Bash:  bash scripts/resume_after_reboot.sh
# It resumes from saved progress and writes to logs/workflow-v3.log. Nothing else was running.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
status=0
.venv/Scripts/python.exe -m bakeoff.workflow_v3 2>&1 | tee logs/workflow-v3.log || status=$?
echo "RESUME_EXIT_$status"
