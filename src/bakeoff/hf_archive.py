"""Publish the specialist weights and cached features to the Hugging Face Hub, and check the upload against release/SHA256SUMS.

    python -m bakeoff.hf_archive stage                 # hard-link the files into STAGE, write the model card and hash list
    python -m bakeoff.hf_archive upload <repo_id>      # create the repo (private) and upload STAGE, resumably
    python -m bakeoff.hf_archive verify <repo_id>      # compare the Hub's SHA-256 of every file with release/SHA256SUMS

Only files for the confirmatory sets are published (Banking77, CLINC150, MASSIVE). Run D's specialists and features, and the
exploratory BANKING77-OOS and MultiDoGO runs, are left out: BANKING77-OOS states no licence for its split, and the MultiDoGO runs
(CDLA-Permissive-1.0) are exploratory. Their hashes stay in release/SHA256SUMS and the code and frozen item lists rebuild them.
"""

import os
import shutil
import sys
from pathlib import Path

from bakeoff.protocol import ROOT

STAGE = Path("D:/System One HF staging")
INCLUDE = [
    "artifacts/confirmatory/specialists",
    "artifacts/confirmatory/features",
    "artifacts/massive-robustness/specialists",
    "artifacts/massive-robustness/features",
    "artifacts/revision/features",
    "artifacts/robustness/banking77",
    "artifacts/robustness/clinc150",
    "artifacts/robustness/massive",
]
EXCLUDE_NAME_PARTS = ("banking77_oos", "multidogo")

CARD = """---
license: mit
base_model: jhu-clsp/ettin-encoder-150m
tags:
- intent-classification
- out-of-scope-detection
- text-classification
---

# Specialist classifiers and cached features for "Matched data, small gaps"

Fine-tuned Ettin 150M encoders (the paper's *specialist*) and the cached features its analyses read, for Banking77, CLINC150 and MASSIVE.
The code, frozen protocols, item lists, saved model replies and analyses are in the paper's release repository
(https://github.com/SafwanKamal/matched-data-small-gaps); this repository holds the files too large for git.
Paper (preprint): https://doi.org/10.5281/zenodo.23448302. Every file here is listed with its SHA-256 in `SHA256SUMS` (and in the release's `release/SHA256SUMS`).

## Layout

Paths mirror the release repository, so the files can be copied into a clone as they are.

| Path | What it holds |
|---|---|
| `artifacts/confirmatory/specialists/<variant>/<set>/seed-<seed>/` | The confirmatory specialists: `encoder/` (weights and tokenizer), `head.pt` (classification head), `metrics.json`, `test_scores.npz` |
| `artifacts/confirmatory/features/` | Cached logits and detector inputs for every confirmatory turn and development item, per set and seed |
| `artifacts/massive-robustness/specialists/`, `features/` | The same for the three further MASSIVE draws |
| `artifacts/revision/features/` | Features for the earlier exploratory analyses (Banking77, CLINC150, MASSIVE) |
| `artifacts/robustness/<set>/seed-<seed>/` | Exploratory seed runs (Banking77, CLINC150, MASSIVE) |

## Training

- Base model `jhu-clsp/ettin-encoder-150m` (revision pinned in `primary_protocol.json`), with a mean-pooled linear classification head, fine-tuned on the
  lower-cased, punctuation-stripped training rows of each set.
- Learning rate and epochs: Banking77 1e-4 for 6 epochs; CLINC150 5e-5 for 3; MASSIVE 2e-4 for 6 (chosen on an earlier 60-label task, not re-swept).
- Seeds 20261007, 20261015 and 20261016. Temperature is fitted on development in-scope items; the scope decision uses a development-selected detector
  (maximum softmax probability, energy, Mahalanobis or kNN) with a threshold weighted by a 20% out-of-scope prior.
- The full rules are in `confirmatory_protocol.json` and `massive_robustness_protocol.json` in the release repository.

## Not included

Run D's specialists and features and the exploratory BANKING77-OOS and MultiDoGO runs are not published here: BANKING77-OOS states no licence for its split, and the MultiDoGO runs (CDLA-Permissive-1.0) are exploratory.
Their hashes are in the release's `release/SHA256SUMS`, and the code rebuilds them from the frozen item lists.

## Licences

The weights are released under MIT, like the base model. The training data carry their own licences: Banking77 CC BY 4.0, CLINC150 CC BY 3.0,
MASSIVE CC BY 4.0. Attribute them when you use these files.

## Citation

```bibtex
@misc{kamal2026matched,
  title     = {Matched data, small gaps: a pre-registered comparison of fine-tuned classifiers, retrieval-prompted language models and a typed-decision model for intent and out-of-scope decisions},
  author    = {Kamal, Safwan},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23448302},
  url       = {https://doi.org/10.5281/zenodo.23448302},
  note      = {Preprint}
}
```

The DOI above always resolves to the latest version; version 1 is https://doi.org/10.5281/zenodo.23448303.
"""


def wanted(rel: str) -> bool:
    return not any(part in rel for part in EXCLUDE_NAME_PARTS) and "__pycache__" not in rel


def listed_hashes() -> dict[str, str]:
    out = {}
    for line in (ROOT / "release" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            h, path = line.split("  ", 1)
            out[path] = h
    return out


def stage() -> None:
    hashes = listed_hashes()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    files, missing = [], []
    for top in INCLUDE:
        for f in sorted((ROOT / top).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(ROOT).as_posix()
            if not wanted(rel):
                continue
            if rel not in hashes:
                missing.append(rel)
                continue
            dst = STAGE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)
            files.append(rel)
    (STAGE / "README.md").write_text(CARD, encoding="utf-8", newline="\n")
    (STAGE / "SHA256SUMS").write_text("".join(f"{hashes[r]}  {r}\n" for r in files), encoding="utf-8", newline="\n")
    size = sum((STAGE / r).stat().st_size for r in files)
    print(f"staged {len(files)} files, {size / 1e9:.2f} GB, in {STAGE}")
    if missing:
        print(f"not in SHA256SUMS, left out: {len(missing)}", *missing[:10], sep="\n  ")


def upload(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=str(STAGE), repo_type="model")
    print("uploaded", repo_id)


def verify(repo_id: str) -> None:
    from huggingface_hub import HfApi

    expected = {}
    for line in (STAGE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, path = line.split("  ", 1)
        expected[path] = h
    info = HfApi().model_info(repo_id, files_metadata=True)
    remote = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    missing = [p for p in expected if p not in remote]
    no_lfs = [p for p in expected if p in remote and remote[p] is None]
    wrong = [p for p in expected if remote.get(p) and remote[p] != expected[p]]
    print(f"{len(expected)} expected, {len(expected) - len(missing)} on the Hub, {len(wrong)} hash mismatches, {len(no_lfs)} stored without LFS hash")
    for p in (missing + wrong)[:20]:
        print("  problem:", p)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage":
        stage()
    elif cmd == "upload":
        upload(sys.argv[2])
    elif cmd == "verify":
        verify(sys.argv[2])
