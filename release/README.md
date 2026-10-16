# Release package: reproducing the paper

This folder accompanies `paper/main.tex` (draft 3).

| File | What it holds |
|---|---|
| `SHA256SUMS` | Hashes of the data, every protocol, every frozen item list and saved model reply, every analysis output, and the specialist weights and features. The weights and features are **not in git** (about 22 GB) and will ship as a separate archive checked against these hashes. |
| `requirements-lock.txt` | The Python environment, as frozen. |
| `ENVIRONMENT.txt` | Python, platform, torch and GPU. |

`src/bakeoff/release_manifest.py` regenerates all three.

## Status of the archive

The specialist weights and features (about 22 GB) are **not yet published**; their hashes are in `SHA256SUMS`. Until the archive ships, the analyses that need specialist features cannot be rerun by a third party; the hosted-model analyses can, from the saved replies in git. The loaders check data hashes only for MASSIVE (`primary_protocol.json`); the other data sets are listed in `SHA256SUMS` and must be checked by hand.

## What is in git

- **Protocols** (repository root), each frozen before any of its requests, with dated amendments and errata:
  - `confirmatory_protocol.json`
  - `retrieval_suppression_protocol.json` (run D)
  - `jev_rerun_protocol.json`
  - `retrieval_mechanism_protocol.json` (run D2)
  - `massive_robustness_protocol.json`
  - `gh_protocol.json` (runs G and H)
- **Frozen items:** `artifacts/<run>/items.json`. Each item list's SHA-256 is recorded inside its protocol.
- **Saved replies** from every hosted model, one JSON line per request with text, tokens, cost, latency and provider: `artifacts/<run>/hosted/`, `jev/` and `replies/`.
- **Analyses:** `artifacts/<run>/analysis.json` and `artifacts/posthoc/*.json`.
- **Code:** `src/bakeoff/`, with one module per protocol.
- **Freeze table:** `release/FREEZE_TABLE.md` lists, for each protocol, its freeze time, its SHA-256 as frozen, its item-list hash and when its replies and analysis were first committed; `release/frozen-protocols/` holds each protocol as first committed. The order can be checked from these files without the git history.

## Regenerating the numbers from saved replies (CPU only, no API)

Each analysis reads only saved replies and cached specialist features. Run from the repository root:

```
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$PWD/.cache/hf"  # the cached datasets live here; offline runs fail without HF_HOME
python -m bakeoff.confirmatory analyze
python -m bakeoff.retrieval_suppression analyze
python -m bakeoff.jev_rerun analyze
python -m bakeoff.retrieval_mechanism analyze
python -m bakeoff.massive_robustness analyze
python -m bakeoff.gh_runs analyze
python -m bakeoff.posthoc_round3
python -m bakeoff.posthoc_round3b
python -m bakeoff.paper_number_check
```

In round 3, a separate check regenerated `confirmatory/analysis.json`, `retrieval-suppression/analysis.json` and `posthoc/round3.json` bit for bit.

The analyses need the specialist features (`artifacts/*/features/*.npz`), which are in the archive. Rebuilding them needs the weights and a GPU:

```
python -m bakeoff.<module> features
```

## Re-running from scratch (costs money)

Each protocol's module has `build`, `train`, `check` and `hosted` stages:

- **Pinned hosts, models and prices** are in each protocol and in the paper's Appendix B.
- **Spend** was $72.08 on OpenRouter before runs G and H. Jev was billed separately by TypeSafe: $4.27 in total for all Jev use in the project (prepaid balance), which cannot be split per call.
- **Variance:** hosted models change over time, and moving one model between two hosts changed 9.5% of its answers. Expect answers to differ somewhat even at temperature 0.

## Data provenance

The data is not redistributed. Each source with its licence:

| Data | Source | Licence |
|---|---|---|
| Banking77 | `legacy-datasets/banking77` | CC BY 4.0 |
| CLINC150 | `clinc/clinc_oos` (plus, revision 155b9c71) | CC BY 3.0 |
| MASSIVE | `data/1.1/en-US.jsonl`, checked against the hash in `primary_protocol.json` | CC BY 4.0 |
| BANKING77-OOS | `github.com/jianguoz/Few-Shot-Intent-Detection`, commit fa955331 | no licence stated |
| CLINC single-domain out-of-scope | `github.com/jianguoz/Few-Shot-Intent-Detection`, commit fa955331 | no licence stated |

**Known data issue:** the published BANKING77-OOS training split mislabels 366 of 5,905 rows. 145 of them are held-out intents. The audit is in `artifacts/posthoc/b77oos_label_audit.json`.
