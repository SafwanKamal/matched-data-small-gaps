#!/usr/bin/env bash
# Build the bakeoff environment inside WSL. The venv lives on the Linux
# filesystem because /mnt/d is slow for many small files.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VENV="${BAKEOFF_VENV:-$HOME/venvs/bakeoff}"
cd "$(dirname "$0")/.."

uv venv --python 3.12 --allow-existing "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$VENV/bin/python" -r requirements.txt pytest

"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
PYTHONPATH=src "$VENV/bin/python" -m pytest tests -q
echo "SETUP_OK"
