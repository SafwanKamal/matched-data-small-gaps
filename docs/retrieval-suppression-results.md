# Run D: retrieval suppression, results (2026-10-16)

## Sources

- **Protocol:** `retrieval_suppression_protocol.json`, frozen before any request, with a dated DeepSeek host amendment; see `release/FREEZE_TABLE.md` for its freeze time, frozen text and item-list hash.
- **Analysis:** `src/bakeoff/retrieval_suppression.py analyze`, writing `artifacts/retrieval-suppression/analysis.json`.

## Setup

- **Requests:** every one was answered and parsed: 0 failures and 0 parse failures over 44,432 calls.
- **Models:** four general models, each on a pinned host with reasoning off:
  - Gemma 4 31B on Novita;
  - Qwen3.5 27B on Alibaba;
  - Llama 3.3 70B on DeepInfra;
  - DeepSeek V4 Pro on StreamLake.
- **Prompt conditions (arms):**
  - zero-shot;
  - retrieval of 20 in-scope examples;
  - the retrieval prompt plus an instruction to answer out of scope when nothing in the options covers the request (the instruction mitigation);
  - retrieval plus four fixed off-topic out-of-scope examples (the rows mitigation).
- **Items:** fresh items on two same-domain splits.
  - **BANKING77-OOS:** 398 in-scope items and 679 held-out-intent items, from 27 held-out intents.
  - **CLINC single-domain:** banking and credit cards, 1,000 in-scope and 700 held-out items, from 10 held-out intents.

## Primary test (H1): does retrieval lower the rejection of held-out-intent requests?

The test is one-sided and Holm-corrected over 8 tests. The table shows the share of held-out requests rejected zero-shot, then with retrieval. The difference is zero-shot minus retrieval, in points.

| Model | BANKING77-OOS | Difference (Holm p) | CLINC single-domain | Difference (Holm p) |
|---|---|---|---|---|
| Gemma 4 31B | 59.5 → 30.3 | +29.2 (.0008) | 42.6 → 47.6 | −5.0 (1.0) |
| Qwen3.5 27B | 82.2 → 58.5 | +23.7 (.0008) | 67.0 → 68.9 | −1.9 (1.0) |
| Llama 3.3 70B | 50.7 → 17.7 | +33.0 (.0008) | 34.4 → 39.4 | −5.0 (1.0) |
| DeepSeek V4 Pro | 60.8 → 21.1 | +39.8 (.0008) | 30.6 → 31.7 | −1.1 (1.0) |

The protocol fixed these sentences in advance:

- **BANKING77-OOS: replicates.** Retrieving in-scope examples lowers the rejection of same-domain unsupported requests for 4 of 4 general models, by 24 to 40 points.
- **CLINC single-domain: does not replicate.** Retrieval lowers that rejection for 0 of 4 models, so the suppression is not general. On these splits, retrieval leaves rejection unchanged or slightly higher.

### What moves together with the drop (read before interpreting)

On BANKING77-OOS the zero-shot arm also rejects many *in-scope* requests:

| Model | In-scope kept, zero-shot | In-scope kept, retrieval | In-scope right, zero-shot | In-scope right, retrieval |
|---|---|---|---|---|
| Gemma 4 31B | 91.7 | 98.2 | 75.9 | 86.7 |
| Qwen3.5 27B | 80.9 | 95.2 | 66.3 | 85.9 |
| Llama 3.3 70B | 89.2 | 99.2 | 60.8 | 87.7 |
| DeepSeek V4 Pro | 88.9 | 99.0 | 71.6 | 88.4 |

On CLINC single-domain, zero-shot already keeps 88–97% of in-scope requests. The pattern fits retrieval moving each model toward accepting more requests in general. It fits a mechanism specific to unsupported requests less well. Retrieval suppresses rejection where zero-shot over-rejects across the board, and does nothing where it does not.

This reading is post hoc. The protocol did not test it.

## Mitigations (H2)

These tests are one-sided and Holm-corrected over 16 tests.

### Instruction mitigation

| Split | Change in rejection vs retrieval | Holm-significant | Share of the drop recovered | In-scope right, change |
|---|---|---|---|---|
| BANKING77-OOS | Gemma +3.0, Qwen +5.0, Llama +14.0, DeepSeek +10.8 | 3 of 4 (all but Gemma) | 10%, 21%, 42%, 27% | −0.25, −0.75, −1.76 (95% CI [−2.98, −0.71]), −0.25 |
| CLINC single-domain | +15 to +27 for all four models | 4 of 4 (Holm p .023–.038) | not defined (there was no drop) | −0.4 to −1.2 points |

- On BANKING77-OOS the instruction recovers less than the 50% of the drop that the "works" criterion requires, for every model. So it **does not work** by the pre-registered criterion.
- On CLINC single-domain, held-out rejection rises well above the zero-shot level (for example, Gemma 47.6 → 70.3).
- **Rows mitigation: no help.** On BANKING77-OOS the change is between −2.7 and +0.9 points. On CLINC single-domain, rejection *falls* by 6 to 8 points.

The protocol fixed this sentence in advance:

- **Mitigation:** neither mitigation recovers half of the suppression without costing more than 2 points of in-scope accuracy.

A secondary observation, unadjusted: a one-sentence instruction is the effective lever. It is cheap and raises rejection of same-domain unsupported requests by 3–27 points at a cost of up to about 2 points of in-scope accuracy.

## Specialist reference

These are the 150M specialists, three seeds, prior-weighted threshold. The specialist's threshold saw 100 held-out-intent validation items.

| Split | Held-out rejected | In-scope right |
|---|---|---|
| BANKING77-OOS | 49.6% | 85.8% |
| CLINC single-domain | 53.7% | 93.6% |

With retrieval, the general models on BANKING77-OOS reject 18–59% of held-out requests, with in-scope right at 86–88%.

## Cost

- **Run D in total:** $5.06. That includes the hosted calls ($4.88), the checks ($0.05) and the Parasail partial ($0.13).
- **Per 1,000 calls, retrieval arm, BANKING77-OOS:**

  | Model | Cost |
  |---|---|
  | Llama 3.3 70B | about $0.08 |
  | Gemma 4 31B | about $0.13 |
  | Qwen3.5 27B | about $0.19 |
  | DeepSeek V4 Pro | about $0.19 (after StreamLake's prompt caching) |

  The per-cell figures are in `analysis.json`, under `models.<name>.<arm>.cost`.
- **Project spend:** $65.19 of the $80 cap, reconciled from saved rows.
