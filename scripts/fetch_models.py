"""Download model snapshots into the project cache on D: and record their revisions. Network only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for repo in sys.argv[1:]:
        print(f"model {repo}", flush=True)
        path = snapshot_download(repo)
        manifest["models"][repo] = {"revision": Path(path).name}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  done {repo} {Path(path).name}", flush=True)
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
