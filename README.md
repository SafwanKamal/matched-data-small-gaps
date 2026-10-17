# Matched data, small gaps

Code, frozen protocols, saved model replies and analyses for the paper *Matched data, small gaps: a pre-registered comparison of fine-tuned classifiers, retrieval-prompted language models and a typed-decision model for intent and out-of-scope decisions* (Safwan Kamal).

- **Paper:** [`paper/main.pdf`](paper/main.pdf), source in [`paper/main.tex`](paper/main.tex).
- **Reproducing the numbers:** [`release/README.md`](release/README.md). The hosted-model analyses rerun on a CPU from the saved replies, with no API calls; analyses that use the specialist also need its weights and features, which are on the Hugging Face Hub at https://huggingface.co/Safka532/matched-data-small-gaps-specialists (see release/README.md).
- **Pre-registration:** [`release/FREEZE_TABLE.md`](release/FREEZE_TABLE.md) gives each protocol's freeze time, its text as frozen (`release/frozen-protocols/`), its item-list hash, and when its replies and analysis were first committed.
- **File hashes:** [`release/SHA256SUMS`](release/SHA256SUMS) covers the data, every protocol, frozen item list, saved reply and analysis output, and the specialist weights and features.

## Layout

| Path | What it holds |
|---|---|
| `*_protocol.json` (repository root) | One protocol per experiment, frozen before any of its requests, with dated amendments and errata. |
| `artifacts/<run>/` | Frozen item lists, saved replies from every hosted model (one JSON line per request with text, tokens, cost, latency and provider) and analysis outputs. |
| `src/bakeoff/` | Code, one module per protocol. |
| `docs/*-results.md`, `docs/posthoc-*.md` | Results write-ups for each run and the post hoc analyses. |
| `docs/figures/` | Figures used in the paper. |
| `scripts/`, `tests/` | Run scripts and unit tests. |

Experiments named in the paper map to protocols as follows: confirmatory runs A, B, C and I (`confirmatory_protocol.json`), run D (`retrieval_suppression_protocol.json`), the Jev re-run (`jev_rerun_protocol.json`), run D2 (`retrieval_mechanism_protocol.json`), the MASSIVE robustness runs (`massive_robustness_protocol.json`) and runs G and H (`gh_protocol.json`). The other protocols are the earlier exploratory record.

## Data

The data set files are not redistributed. The frozen item lists contain the utterances used, all verbatim from Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0) or MASSIVE (CC BY 4.0); `release/README.md` lists each source, revision and licence.

## Licence

Code and project files: MIT License (`LICENSE`). Data-derived content stays under its source's licence (see `release/README.md`). The specialist weights on the Hugging Face Hub are also MIT.

## Setup

Python 3.12. `release/requirements-lock.txt` is the frozen environment and `release/ENVIRONMENT.txt` records the platform and GPU. API keys, if you rerun paid stages, are read from a local `.env` file that is never committed.
