#!/usr/bin/env bash
# Wait for the extra-seed chain to end, then score the 4B index readout on the wider suite.
# The two cannot share the 12 GB card, so this never starts while a trainer is alive.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
until grep -q "SEEDS_EXIT_" logs/seeds.log 2>/dev/null; do sleep 120; done
echo "=== $(date -u +%H:%M:%SZ) seeds chain ended: $(grep -o 'SEEDS_EXIT_[0-9]*' logs/seeds.log | tail -1)"
status=0
.venv/Scripts/python.exe -m bakeoff.suite_run llm qwen3-4b-index_prior || status=$?
echo "AFTER_SEEDS_EXIT_$status"
