#!/usr/bin/env bash
# After the variety curve: workflow v1 with the GPU workers, then the MASSIVE extra seeds. One GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
until grep -q "VARIETY_DONE\|Traceback" logs/variety.log 2>/dev/null; do sleep 30; done
status=0
for module in workflow primary_seeds; do
  echo "=== $(date -u +%H:%M:%SZ) bakeoff.$module"
  .venv/Scripts/python.exe -m "bakeoff.$module" || { status=$?; echo "STEP_FAILED bakeoff.$module exit $status"; }
done
echo "CHAIN_EXIT_$status"
