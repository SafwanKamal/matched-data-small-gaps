# Release package: reproducing the paper

This folder accompanies `paper/main.tex`.

| File | What it holds |
|---|---|
| `SHA256SUMS` | Hashes of the data, the ten protocols that govern the paper's runs, every frozen item list and saved model reply, every analysis output, and the specialist weights and features. The weights and features are **not in git** (about 22 GB) and will ship as a separate archive checked against these hashes. |
| `requirements-lock.txt` | The Python environment, as frozen. |
| `ENVIRONMENT.txt` | Python, platform, torch and GPU. |

`src/bakeoff/release_manifest.py` regenerates all three.

To check the files: `grep -v '^#' release/SHA256SUMS | sha256sum -c`. In the git repository alone, 647 of the 1,019 entries resolve. The other 372 are the data set files (not redistributed) and the weights and features. They report as unreadable until they are in place. After the Hugging Face download, only the data sets and the unpublished run D and BANKING77-OOS/MultiDoGO files remain.

## Specialist weights and features

The specialist weights and cached features for Banking77, CLINC150 and MASSIVE (222 files, 17.9 GB) are on the Hugging Face Hub at https://huggingface.co/Safka532/matched-data-small-gaps-specialists, with the same paths as in this repository and the same SHA-256 values as `SHA256SUMS`. Copy them into a clone (for example with `hf download Safka532/matched-data-small-gaps-specialists --local-dir .`) and every analysis reruns on a CPU.

Run D's specialists and features are not in the archive. Neither are the exploratory BANKING77-OOS and MultiDoGO runs. BANKING77-OOS states no licence for its split, and the MultiDoGO runs (CDLA-Permissive-1.0) are exploratory. Their hashes remain in `SHA256SUMS`. Rebuilding them needs the original BANKING77-OOS files (github.com/jianguoz/Few-Shot-Intent-Detection, commit fa955331, `data/banking77-oos`) and a GPU. `python -m bakeoff.hf_archive` stages, uploads and verifies the archive.

**Per-turn specialist scores.** You do not need the archive to check the specialist side. `artifacts/confirmatory/specialist-scores/`, `artifacts/massive-robustness/specialist-scores/` and `artifacts/retrieval-suppression/specialist-scores/` hold the specialist's per-turn scores (30 files, 26 MB). Each file has the logits of the development and test parts, the gold labels, and the Mahalanobis and kNN detector scores. The confirmatory files also have run C's kNN vote. When the cached features are absent, the loaders read these files instead. The analyses then give byte-identical outputs. Run D is included, so its analysis reruns without its specialists. `python -m bakeoff.specialist_scores` writes these files from the features.

The loaders check data hashes only for MASSIVE (`primary_protocol.json`); the other data sets are listed in `SHA256SUMS` and must be checked by hand.

## What is in git

- **Protocols** (repository root), each frozen before any of its requests, with dated amendments and errata:
  - `confirmatory_protocol.json`
  - `retrieval_suppression_protocol.json` (run D)
  - `jev_rerun_protocol.json`
  - `retrieval_mechanism_protocol.json` (run D2)
  - `massive_robustness_protocol.json`
  - `gh_protocol.json` (runs G and H)
  - `option_count_protocol.json` (run E)
  - `reasoning_on_protocol.json` (run R)
- **Frozen items:** `artifacts/<run>/items.json`. Each item list's SHA-256 is recorded inside its protocol.
- **Saved replies** from every hosted model, one JSON line per request with text, tokens, cost, latency and provider: `artifacts/<run>/hosted/`, `jev/` and `replies/`.
- **Analyses:** `artifacts/<run>/analysis.json` and `artifacts/posthoc/*.json`.
- **Provider records:** `artifacts/provider-log/` is OpenRouter's own request log for the project's key (server timestamps, model, provider, tokens, cost; imported with `python -m bakeoff.provider_log`), and `artifacts/jev-cost/` the TypeSafe billing record.
- **Code:** `src/bakeoff/`, with one module per protocol.
- **Freeze table:** `release/FREEZE_TABLE.md` lists, for each protocol, its freeze time, its SHA-256 as frozen, its item-list hash and when its replies and analysis were first committed; `release/frozen-protocols/` holds each protocol as first committed. The order can be checked from these files without the git history.

## Regenerating the numbers from saved replies (CPU only, no API)

Each analysis reads saved replies and the specialist's per-turn scores (or its cached features, when they are present). Run from the repository root:

```
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$PWD/.cache/hf"  # a cache holding the external data below; this repository does not include one
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
python -m bakeoff.posthoc_round6
python -m bakeoff.posthoc_round7
python -m bakeoff.option_count analyze
python -m bakeoff.reasoning_on analyze
python -m bakeoff.b77oos_full_audit
python -m bakeoff.paper_number_check
```

In round 3, a separate check regenerated `confirmatory/analysis.json`, `retrieval-suppression/analysis.json` and `posthoc/round3.json` bit for bit.

Rebuilding the specialist features (`artifacts/*/features/*.npz`) needs the weights and a GPU:

```
python -m bakeoff.<module> features
```

**What runs from this repository alone.** Offline, with an empty Hugging Face cache, eleven of the fifteen commands above run from this repository alone: `confirmatory`, `retrieval_suppression`, `jev_rerun`, `retrieval_mechanism`, `massive_robustness`, `option_count` and `reasoning_on` (all `analyze`), and `posthoc_round5`, `posthoc_round6`, `posthoc_round7` and `paper_number_check`. The other four need external data, which we do not redistribute. Fetch it once with network access, into the cache that `HF_HOME` points to:

| Data | Source and revision | Needed by |
|---|---|---|
| Banking77 | `legacy-datasets/banking77`, revision `f54121560de48f2852f90be299010d1d6dc612ec` | `gh_runs analyze`, `posthoc_round3b`, `b77oos_full_audit` |
| SST-2 | `nyu-mll/glue`, subset `sst2`, revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | `gh_runs analyze` (loaded with Banking77 by `bakeoff.data`) |
| AG News | `fancyzhx/ag_news`, revision `eb185aade064a813bc0b7f42de02595523103ca4` | `gh_runs analyze` (loaded with Banking77 by `bakeoff.data`) |
| all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2`, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` | `posthoc_round3`, and run E's `build` stage |
| MASSIVE 1.1 | `amazon-massive-dataset-1.1.tar.gz` from amazon-massive-nlu-dataset.s3.amazonaws.com, unpacked so that `data/1.1/data/en-US.jsonl` exists (hash in `primary_protocol.json`) | `posthoc_round4` |
| BANKING77-OOS | github.com/jianguoz/Few-Shot-Intent-Detection, commit fa955331, `data/banking77-oos` | `b77oos_full_audit`, and run D's `build` and `train` stages |
| CLINC150 | `clinc/clinc_oos`, config `plus`, revision `155b9c710419136e17307b80d0a13e68cd46b4ec` | the `build` stages of runs D2 and E |

The `build`, `train` and `hosted` stages also need these. The analyses above do not need them, unless the table says so.

**References to working notes.** Some protocols name internal working notes (for example `docs/HANDOFF.md` or files under `docs/reviews/`) in their `role` or `found_by` fields. Those notes are not part of this release; each protocol's rules, items and amendments are complete without them.

## Re-running from scratch (costs money)

Each protocol's module has `build`, `train`, `check` and `hosted` stages:

- **Pinned hosts, models and prices** are in each protocol and in the paper's Appendix B.
- **Spend** on OpenRouter was $74.27 by OpenRouter's own log (`artifacts/provider-log/`; $74.50 reconstructed from the saved rows). Jev was billed separately by TypeSafe: $4.27 in total for all Jev use in the project (one $5.00 credit with $0.73 left; `artifacts/jev-cost/`), which cannot be split per call.
- **Variance:** hosted models change over time, and moving one model between two hosts changed 9.5% of its answers. Expect answers to differ somewhat even at temperature 0.

## Data provenance

The data set files are not redistributed. The frozen item lists and session files (`artifacts/*/items.json`, `artifacts/*/sessions*.json`) and the BANKING77-OOS label audit (`artifacts/posthoc/b77oos_full_audit.csv`) do contain the utterances that were used, and every one of them was checked against the original sets. Each is verbatim, up to case and punctuation, from Banking77, CLINC150 or MASSIVE, including the BANKING77-OOS and CLINC single-domain items. They are redistributed under those sets' licences, with attribution below. Each source with its licence:

| Data | Source | Licence |
|---|---|---|
| Banking77 | `legacy-datasets/banking77` | CC BY 4.0 |
| CLINC150 | `clinc/clinc_oos` (plus, revision 155b9c71) | CC BY 3.0 |
| MASSIVE | `data/1.1/en-US.jsonl`, checked against the hash in `primary_protocol.json` | CC BY 4.0 |
| BANKING77-OOS | `github.com/jianguoz/Few-Shot-Intent-Detection`, commit fa955331 | no licence stated for the split; its utterances are Banking77's |
| CLINC single-domain out-of-scope | `github.com/jianguoz/Few-Shot-Intent-Detection`, commit fa955331 | no licence stated for the split; its utterances are CLINC150's |
| MultiDoGO (exploratory only) | `github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset` | CDLA-Permissive-1.0; only model replies and scores are released, no utterances |

## Licence

The code and the other files written for this project are released under the MIT License (`LICENSE` at the repository root). Data-derived content stays under its source's licence, as listed above.

**Known data issue:** the published BANKING77-OOS training split mislabels 366 of 5,905 rows. 145 of them are held-out intents. The audit is in `artifacts/posthoc/b77oos_label_audit.json`.
