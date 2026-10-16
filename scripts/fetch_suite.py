"""Download the open 4B baseline and the wider dataset suite into the project cache on D:.

Records the commit of every snapshot in data_manifest.json so later protocols can pin revisions.
Network only; does not touch the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifest.json"

MODELS = ["Qwen/Qwen3-4B"]
DATASETS = {
    "clinc/clinc_oos": {"config": "plus"},
    "dair-ai/emotion": {"config": "split"},
    "community-datasets/yahoo_answers_topics": {"config": None},
    "takala/financial_phrasebank": {"config": "sentences_allagree"},
    "facebook/anli": {"config": None},
    "google/boolq": {"config": None},
    "allenai/ai2_arc": {"config": "ARC-Challenge"},
    "cais/mmlu": {"config": "all"},
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"models": {}, "datasets": {}}
    for repo in MODELS:
        if repo in manifest["models"]:
            continue
        print(f"model {repo}", flush=True)
        path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"])
        manifest["models"][repo] = {"revision": Path(path).name}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    from datasets import load_dataset

    for repo, spec in DATASETS.items():
        if manifest["datasets"].get(repo, {}).get("loads"):
            continue
        print(f"dataset {repo}", flush=True)
        entry = {"config": spec["config"]}
        try:
            path = snapshot_download(repo, repo_type="dataset")
            entry["revision"] = Path(path).name
            loaded = load_dataset(repo, spec["config"], revision=entry["revision"]) if spec["config"] else load_dataset(repo, revision=entry["revision"])
            entry["splits"] = {name: len(split) for name, split in loaded.items()}
            entry["features"] = {name: str(feature)[:160] for name, feature in next(iter(loaded.values())).features.items()}
            entry["loads"] = True
        except Exception as error:  # recorded so the suite can be adjusted, not hidden
            entry["loads"] = False
            entry["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        manifest["datasets"][repo] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  {json.dumps({k: v for k, v in entry.items() if k != 'features'})}", flush=True)
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
