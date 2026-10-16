#!/usr/bin/env bash
# CLINC150 third domain (workflow_clinc_protocol.json).
# Now: collect the hosted models' zero-shot replies (API only).
# After the size sweep leaves the GPU: train the two specialists, score the MASSIVE decoder, then run the rest.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
log=logs/workflow-clinc.log
sweeps=$(grep -c SIZE_SWEEP_EXIT_ logs/size-sweep.log)
.venv/Scripts/python.exe -m bakeoff.workflow_clinc collect >> "$log" 2>&1
echo "CLINC_COLLECT_EXIT_$?" >> "$log"
until [ "$(grep -c SIZE_SWEEP_EXIT_ logs/size-sweep.log)" -gt "$sweeps" ]; do sleep 60; done
.venv/Scripts/python.exe -m bakeoff.workflow_clinc train >> "$log" 2>&1
echo "CLINC_TRAIN_EXIT_$?" >> "$log"
.venv/Scripts/python.exe -m bakeoff.workflow_massive decoder >> logs/workflow-massive.log 2>&1
echo "WORKFLOW_MASSIVE_DECODER_EXIT_$?" >> logs/workflow-massive.log
.venv/Scripts/python.exe -m bakeoff.workflow_clinc all >> "$log" 2>&1
echo "WORKFLOW_CLINC_EXIT_$?" >> "$log"
