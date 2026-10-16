#!/usr/bin/env bash
# Rerun every finetune after the float32-weights fix, in order, on Windows (Git Bash).
# Finished jobs are skipped by each trainer, so the same command resumes a stopped run.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
status=0
for step in "train_ettin" "encoder_latency" "train_score" "primary probe" "primary finetune" "train_ettin_decoder"; do
  echo "=== $(date -u +%H:%M:%SZ) bakeoff.$step"
  # shellcheck disable=SC2086
  .venv/Scripts/python.exe -m bakeoff.$step || { status=$?; echo "STEP_FAILED bakeoff.$step exit $status"; }
done
echo "RERUN_EXIT_$status"
exit "$status"
