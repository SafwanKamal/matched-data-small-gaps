# Runs G and H: results (2026-10-16)

## Sources and execution

- **Protocol:** `gh_protocol.json`, frozen at `03f0963` before any request.
- **Code:** `src/bakeoff/gh_runs.py`.
- **Numbers:** `artifacts/gh/analysis.json`.
- **Status:** sensitivity and descriptive analyses, outside every confirmatory family.
- **Requests:** every one was answered.
- **Host latency** (corrected 2026-10-16 after the round 4 review): Gemma's host (Novita) had slow stretches early in the run, but the saved latencies give a median of 1.07 s for G and 1.35 s for H. 2.3% of G calls took more than 5 s, and 25% of H calls. An earlier draft of this file said "medians of 7–10 s", which came from the first few dozen rows only.
- **Spend:** $1.05. The project total is **$73.13** of $80, reconciled from saved rows.

## G: order of the retrieved examples (Gemma 4 31B)

Weighted turn accuracy (%) under each order of the same 20 examples, with specialist − Gemma under that order (95% CI):

| Set | A (as run, least similar first) | G1 (most similar first) | G2 (random order) | Range | Specialist − Gemma under A / G1 / G2 |
|---|---|---|---|---|---|
| CLINC150 | 95.03 | 95.04 | 94.88 | **0.16 points** | −1.89 / −1.90 / −1.75 (each p ≤ .004) |
| MASSIVE | 73.84 | 74.25 | 74.08 | **0.41 points** | +4.42 / +4.01 / +4.18 (none significant) |

**Answers that changed against A:**

| Set | G1 | G2 |
|---|---|---|
| CLINC150 | 0.8% | 0.6% |
| MASSIVE | 4.5% | 3.3% |

**Readout stated in advance:** the range is below 1 point on both sets. So the reported differences between systems are **not** within the general model's sensitivity to example order. Order matters far less than the host does: moving DeepSeek between hosts changed 9.5% of its answers.

## H: contamination probe (completion of the second half of public test utterances)

| Model | Set | Exact completions | Token F1 with own ending | Mean F1 with same-intent endings | Own − sibling (points, p) |
|---|---|---|---|---|---|
| Gemma 4 31B | Banking77 | 0.5% | 0.179 | 0.077 | +10.2 (p = .0001) |
| Gemma 4 31B | CLINC150 | 0.5% | 0.139 | 0.070 | +6.9 (p = .0001) |
| Qwen3.5 27B | Banking77 | 0.0% | 0.074 | 0.067 | +0.7 (p = .13) |
| Qwen3.5 27B | CLINC150 | 0.0% | 0.048 | 0.048 | +0.0 (p = .89) |

**Readout stated in advance:** memorisation is indicated only if the own-minus-sibling difference is significant **and** exact matches exceed 5%. That is not met for either model, so the probe finds no evidence of verbatim memorisation.

- **Gemma's own-ending advantage** is consistent with an utterance's first half predicting its second half. It does not establish contamination either way.
- **Qwen refused 350 of 400 requests** ("I cannot complete this text exactly as it appears in its source"), so its rows say little.
- **A negative probe does not rule out contamination.**
