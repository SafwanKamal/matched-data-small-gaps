# Release package: reproducing the paper

This folder accompanies `paper/main.tex`.

| File | What it holds |
|---|---|
| `SHA256SUMS` | Hashes of the data, every protocol, every frozen item list and saved model reply, every analysis output, and the specialist weights and features. The weights and features are **not in git** (about 22 GB) and will ship as a separate archive checked against these hashes. |
| `requirements-lock.txt` | The Python environment, as frozen. |
| `ENVIRONMENT.txt` | Python, platform, torch and GPU. |

`src/bakeoff/release_manifest.py` regenerates all three.

To check the files: `grep -v '^#' release/SHA256SUMS | sha256sum -c`. In the git repository alone, 422 entries resolve; the other 369 are the data sets (not redistributed) and the weights and features, which report as unreadable until they are in place; after the Hugging Face download, only the data sets and the unpublished run D and BANKING77-OOS/MultiDoGO files remain.

## Specialist weights and features

The specialist weights and cached features for Banking77, CLINC150 and MASSIVE (222 files, 17.9 GB) are on the Hugging Face Hub at https://huggingface.co/Safka532/matched-data-small-gaps-specialists, with the same paths as in this repository and the same SHA-256 values as `SHA256SUMS`. Copy them into a clone (for example with `hf download Safka532/matched-data-small-gaps-specialists --local-dir .`) and every analysis reruns on a CPU.

Run D's specialists and features, and the exploratory BANKING77-OOS and MultiDoGO runs, are not published, because those data sets state no licence. Their hashes remain in `SHA256SUMS`; the `train` and `features` stages rebuild them from the frozen item lists, and until then the analyses that read them can be checked only through the saved outputs. `python -m bakeoff.hf_archive` stages, uploads and verifies the archive.

The loaders check data hashes only for MASSIVE (`primary_protocol.json`); the other data sets are listed in `SHA256SUMS` and must be checked by hand.

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
- **Provider records:** `artifacts/provider-log/` is OpenRouter's own request log for the project's key (server timestamps, model, provider, tokens, cost; imported with `python -m bakeoff.provider_log`), and `artifacts/jev-cost/` the TypeSafe billing record.
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
python -m bakeoff.posthoc_round4
python -m bakeoff.posthoc_round5
python -m bakeoff.paper_number_check
```

In round 3, a separate check regenerated `confirmatory/analysis.json`, `retrieval-suppression/analysis.json` and `posthoc/round3.json` bit for bit.

The analyses need the specialist features (`artifacts/*/features/*.npz`), which are in the archive. Rebuilding them needs the weights and a GPU:

```
python -m bakeoff.<module> features
```

**Extra cached data.** Modules that load Banking77 through `bakeoff.data` (including `gh_runs analyze`) fetch SST-2 (`nyu-mll/glue`, subset `sst2`, revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`) and AG News (`fancyzhx/ag_news`, revision `eb185aade064a813bc0b7f42de02595523103ca4`) with it, so both must be in the Hugging Face cache for offline runs.

**References to working notes.** Some protocols name internal working notes (for example `docs/HANDOFF.md` or files under `docs/reviews/`) in their `role` or `found_by` fields. Those notes are not part of this release; each protocol's rules, items and amendments are complete without them.

## Re-running from scratch (costs money)

Each protocol's module has `build`, `train`, `check` and `hosted` stages:

- **Pinned hosts, models and prices** are in each protocol and in the paper's Appendix B.
- **Spend** on OpenRouter was $72.90 by OpenRouter's own log (`artifacts/provider-log/`; $73.13 reconstructed from the saved rows). Jev was billed separately by TypeSafe: $4.27 in total for all Jev use in the project (one $5.00 credit with $0.73 left; `artifacts/jev-cost/`), which cannot be split per call.
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
