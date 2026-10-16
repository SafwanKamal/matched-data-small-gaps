#!/usr/bin/env bash
# Run the bakeoff natively on Windows (Git Bash) with the project-local venv
# and model cache. Arguments pass through to bakeoff.run.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$(cygpath -w "$PWD/.cache/hf")"
export PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1
status=0
.venv/Scripts/python.exe -m bakeoff.run "$@" || status=$?
echo "RUN_EXIT_$status"
exit "$status"
