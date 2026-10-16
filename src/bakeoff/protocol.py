"""Load protocol.json from the repository root."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_protocol(path: Path | None = None) -> dict:
    protocol_path = path or (ROOT / "protocol.json")
    return json.loads(protocol_path.read_text())
