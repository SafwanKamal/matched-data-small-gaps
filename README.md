# Matched data, small gaps

Code, frozen protocols, saved model replies and analyses for the paper *Matched data, small gaps: a pre-registered comparison of fine-tuned classifiers, retrieval-prompted language models and a typed-decision model for intent and out-of-scope decisions* (Safwan Kamal).

- **Paper:** [`paper/main.pdf`](paper/main.pdf), source in [`paper/main.tex`](paper/main.tex).
- **Reproducing the numbers:** [`release/README.md`](release/README.md). Every analysis reruns on a CPU from the saved replies; no API calls are needed.
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

The data sets are not redistributed. `release/README.md` lists each source, revision and licence.

## Setup

Python 3.12. `release/requirements-lock.txt` is the frozen environment and `release/ENVIRONMENT.txt` records the platform and GPU. API keys, if you rerun paid stages, are read from a local `.env` file that is never committed.
