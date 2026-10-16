#!/usr/bin/env bash
# Repeat the Program 2 recipes under the extra seeds. Encoders first because they are quick.
# Finished jobs are skipped, so the same command resumes a stopped run.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
status=0
for module in train_ettin train_ettin_decoder; do
  for seed in 20261008 20261009; do
    echo "=== $(date -u +%H:%M:%SZ) bakeoff.$module --seed $seed"
    .venv/Scripts/python.exe -m "bakeoff.$module" --seed "$seed" || { status=$?; echo "STEP_FAILED bakeoff.$module seed $seed exit $status"; }
  done
done
echo "SEEDS_EXIT_$status"
exit "$status"
