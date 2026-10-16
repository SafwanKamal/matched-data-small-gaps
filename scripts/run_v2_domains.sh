#!/usr/bin/env bash
# Local 9B orchestrator on MASSIVE and CLINC150 (workflow_v2_domains_protocol.json). GPU, no API.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe -m bakeoff.workflow_v2_domains "$@" >> logs/v2-domains.log 2>&1
echo "V2_DOMAINS_EXIT_$?" >> logs/v2-domains.log
