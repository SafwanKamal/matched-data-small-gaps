#!/usr/bin/env bash
# GPU chain, one job at a time, no API call:
#   1. workflows v1 to v3 rerun with the specialists trained at the dev-selected rate (workflow_refresh_protocol.json)
#   2. the training-data sweep on MASSIVE and CLINC150 (data_fraction_domains_protocol.json)
# Every step skips finished work, so the script can be rerun after a stop.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_refresh all >> logs/refresh.log 2>&1
echo "REFRESH_EXIT_$?" >> logs/refresh.log
for domain in massive clinc150; do
  .venv/Scripts/python.exe -m bakeoff.data_fraction_domains train "$domain" >> logs/data-fraction-domains.log 2>&1
  .venv/Scripts/python.exe -m bakeoff.data_fraction_domains join "$domain" >> logs/data-fraction-domains.log 2>&1
done
echo "DOMAINS_EXIT_$?" >> logs/data-fraction-domains.log
