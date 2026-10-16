# Run D2: retrieval mechanism, results (2026-10-16)

## Sources

- **Protocol:** `retrieval_mechanism_protocol.json`, frozen at `a5802f7` before any request. The execution note on parallel processes is at `f9c6c1b`.
- **Code:** `src/bakeoff/retrieval_mechanism.py`.
- **Numbers:** `artifacts/retrieval-mechanism/analysis.json`.

## Execution

- **Requests:** every one was answered and parsed, with 0 failures. That covers the four general models (about 39,400 calls) and Jev (about 10,000).
- **Spend:** $3.18 for D2, including the check. The project total is **$68.36** of the $80 cap, reconciled from saved rows.
- **Host latency** (corrected 2026-10-16 after the round 4 review): Gemma's host (Novita) had a slow stretch during its last two cells. Over all of D2 the saved latencies give a median of 0.93 s, and 4.7% of calls took more than 5 s. The host was not changed.

## Items

- **BANKING77-OOS:** fresh, label-clean validation items: 500 in-scope and 388 held-out from 23 held-out intents. The retrieval pool had all 366 mislabelled rows removed, so no held-out intent leaks.
- **CLINC single-domain:** validation items, 1,000 in-scope and 575 held-out. The in-scope items had been run D's specialist development data (disclosed before the freeze).

## BANKING77-OOS

Each cell shows held-out rejected / in-scope kept, in %.

| Model | Zero-shot | Retrieval | Random examples | Rule after examples |
|---|---|---|---|---|
| Gemma 4 31B | 74.0 / 94.0 | 50.0 / 98.6 | 69.1 / 94.6 | 51.5 / 98.6 |
| Qwen3.5 27B | 91.5 / 85.0 | 72.7 / 96.8 | 90.2 / 84.6 | 83.2 / 94.8 |
| Llama 3.3 70B | 59.8 / 92.2 | 36.1 / 100.0 | 49.5 / 97.0 | 54.9 / 99.4 |
| DeepSeek V4 Pro | 68.0 / 89.2 | 41.8 / 99.4 | 62.6 / 95.0 | 49.0 / 99.0 |
| Jev 1.13.0 | 78.4 / 91.4 | 48.7 / 99.8 | 67.5 / 97.4 | 57.5 / 98.6 |

## Pre-registered tests

Each hypothesis is tested across five models with Holm correction and clustered by held-out intent. The protocol fixed the outcome sentences in advance.

| Hypothesis | Gemma | Qwen 27B | Llama 70B | DeepSeek | Jev |
|---|---|---|---|---|---|
| **H1** zero-shot − retrieval rejection (one-sided) | +24.0 (.0006) | +18.8 (.0006) | +23.7 (.0005) | +26.3 (.0005) | +29.6 (.0006) |
| **H2** random − retrieval rejection (two-sided) | +19.1 (.002) | +17.5 (.001) | +13.4 (.004) | +20.9 (.002) | +18.8 (.0005) |
| **H3** rule-after − retrieval rejection (one-sided) | +1.6 (.13) | +10.6 (.0008) | +18.8 (.0005) | +7.2 (.007) | +8.8 (.0008) |
| **H4** CLINC (retrieval − zero-shot), 20 options minus 10 options (two-sided) | −15.3 (.84) | −2.1 (1.0) | **−29.0 (.04)** | −16.5 (.22) | −4.5 (1.0) |

*Holm-corrected p-values are in parentheses.*

1. **H1 replicates**, 5 of 5 models in the same direction. On fresh, label-clean BANKING77-OOS items with a clean pool, retrieval lowers rejection of unsupported same-domain requests for 5 of 5 models, by 19 to 30 points.
2. **H2: random examples leave rejection higher.** Random in-scope examples shift the models less than retrieved ones. Relevant examples drive the acceptance shift: random examples move rejection by only 1–11 points from zero-shot, against 19–30 for retrieved ones.
3. **H3: significant for 4 of 5.** Restating the scope rule after the examples recovers 7 to 19 points of rejection for Qwen, Llama, DeepSeek and Jev, but not for Gemma (+1.6, n.s.). Part of the shift comes from separating the rule from the query.
4. **H4: not significant for 4 of 5.** Doubling the options does not reliably change retrieval's effect on CLINC single-domain. The direction is the same for all five models, though: with 20 options, retrieval lowers rejection, for example Gemma 43.0 → 29.2, Llama 45.2 → 19.8 and DeepSeek 33.7 → 15.8. The effect is significant for Llama only. Option count may contribute, but we cannot establish it.

## Mechanism measures (descriptive)

**BANKING77-OOS, zero-shot → retrieval.** Every model moves strongly toward acceptance, and discrimination stays level or improves.

| Model | Criterion c | d′ |
|---|---|---|
| Gemma | +0.46 → +1.10 | 2.20 → 2.20 |
| Qwen | −0.17 → +0.62 | 2.41 → 2.46 |
| Llama | +0.59 → +1.72 | 1.67 → 2.73 |
| DeepSeek | +0.38 → +1.36 | 1.71 → 2.30 |
| Jev (AUROC) | — | 0.934 → 0.965 |

Higher c means more accepting. The general models' d′ assumes equal variances.

**CLINC single-domain:** Jev's AUROC rises from 0.88 to 0.99 with retrieval at either option count.

## What this changes

- **The acceptance shift is now established on fresh, leak-free items for five models,** with a pre-registered cause: the *relevance* of the retrieved examples. Their mere presence moves the models only a little, and moving the scope rule after the examples partly undoes the shift.
- **Why CLINC single-domain shows no shift at 10 options remains unexplained.** Option count points in the predicted direction for all five models, but is significant for only one.
