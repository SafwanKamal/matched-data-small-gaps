#!/usr/bin/env bash
# revision_free_protocol.json, one stage of bakeoff.revision_free per call. Finished work is cached and skipped on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
for stage in "$@"; do
  .venv/Scripts/python.exe -m bakeoff.revision_free "$stage" >> logs/revision-$stage.log 2>&1
  echo "REVISION_${stage}_EXIT_$?" >> logs/revision-$stage.log
done
