#!/usr/bin/env bash
# Run the bakeoff inside WSL on the local GPU. Arguments pass through to
# bakeoff.run, e.g. scripts/run_wsl.sh --freeze-splits
set -euo pipefail
VENV="${BAKEOFF_VENV:-$HOME/venvs/bakeoff}"
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
status=0
PYTHONPATH=src "$VENV/bin/python" -m bakeoff.run "$@" || status=$?
echo "RUN_EXIT_$status"
exit "$status"
