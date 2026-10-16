# MASSIVE robustness: results (2026-10-16)

## Sources and execution

- **Protocol:** `massive_robustness_protocol.json`, frozen before any training or request; see `release/FREEZE_TABLE.md` for its freeze time, frozen text and item-list hash.
- **Code:** `src/bakeoff/massive_robustness.py`.
- **Numbers:** `artifacts/massive-robustness/analysis.json`.
- **Status:** a **sensitivity analysis**. Items reuse MASSIVE test utterances, so these results are not part of the confirmatory family.
- **Requests:** every one was answered and parsed (0 failures).
- **Specialists:** nine trained, three seeds for each of three draws.
- **Spend:** $3.72 including the check. The project total is **$72.08** of $80, reconciled from saved rows.

## Draws

| Draw | Test out-of-scope domains (T) | Rows-source domains (R) | Labels kept |
|---|---|---|---|
| 1 | alarm, play, social | email, lists, music | 39 |
| 2 | calendar, iot, play | cooking, music, recommendation | 34 |
| 3 | audio, takeaway, weather | iot, news, play | 38 |

Each draw has 1,000 in-scope and 250 out-of-scope items.

## Results

Each cell shows weighted turn accuracy / held-out rejected, in %.

| System | Draw 1 | Draw 2 | Draw 3 | Original (confirmatory) |
|---|---|---|---|---|
| Specialist | 83.1 / 80.0 | 83.1 / 72.8 | 82.3 / 90.8 | 78.3 / 80.7 |
| Gemma, no rows | 84.1 / 68.0 | 84.7 / 62.8 | 82.0 / 61.2 | 73.8 / 23.0 |
| Gemma, same-source rows | 89.7 / 94.8 | 89.8 / 90.4 | 88.5 / 95.6 | (87.2 / 90.5, run C) |
| Gemma, cross-source rows | 86.1 / 76.8 | 85.5 / 68.8 | 82.3 / 66.4 | — |
| Gemma, instruction | 85.5 / 82.4 | 85.8 / 78.4 | 84.0 / 80.8 | 75.3 / 42.4 |
| Qwen 27B, no rows | 86.5 / 82.8 | 85.3 / 72.8 | 84.2 / 81.6 | 80.8 / 68.3 |
| Jev, no rows | 84.6 / 70.4 | 83.9 / 56.4 | 78.5 / 47.6 | 77.8 / 44.0 |

## Pre-registered tests

The protocol fixed the outcome sentences in advance.

**M1, specialist − Gemma (no rows):**

| Draw | Difference (95% CI) | p |
|---|---|---|
| 1 | −0.96 [−3.29, +1.14] | .39 |
| 2 | −1.65 [−3.78, +0.32] | .24 |
| 3 | +0.27 [−1.67, +2.21] | .87 |

The confirmatory sign holds in 1 of 3 draws. **Outcome sentence:** on further held-out-domain draws the specialist-minus-Gemma difference on MASSIVE changes sign (−1.0, −1.7, +0.3), so the confirmatory MASSIVE result depends on which domains are held out.
- Against Qwen the specialist is behind on every draw: −3.4, −2.2 and −1.9.
- Gemma's 23% rejection on the original draw is atypical. On the new draws it rejects 61–68%, and it still fails on sibling domains: alarm 6%, calendar 38%, audio 42%.

**M3, same-source rows minus no rows** (one-sided, Holm over 6):

| Model | Draw 1 | Draw 2 | Draw 3 |
|---|---|---|---|
| Gemma | +5.6 | +5.1 | +6.5 |
| Qwen | +2.3 | +4.0 | +3.3 |

All six have Holm p = .0003. The rows effect replicates.

**M4, cross-source rows minus no rows** (one-sided, Holm over 6):

| Model | Draw 1 | Draw 2 | Draw 3 |
|---|---|---|---|
| Gemma | **+2.0 (Holm .0003)** | +0.8 (.35) | +0.3 (.98) |
| Qwen | +0.3 (.98) | +0.3 (.98) | +0.2 (.98) |

Only 1 of 6 is significant. **Outcome sentence:** labelled examples of other unsupported domains do not help; the +13-point rows effect of the confirmatory run required examples of the very domains being tested. The one exception is Gemma on draw 1, where the rows came from email, lists and music and the test domains were alarm, play and social.

**M5, instruction minus no rows** (one-sided, Holm over 8):

| Model | Draw 1 | Draw 2 | Draw 3 | Original |
|---|---|---|---|---|
| Gemma | +1.4 | +1.0 | +2.0 | +1.5 |
| Qwen | −0.6 | −0.1 | +0.1 | +1.1 |

All Holm p ≥ .11. **Outcome sentence:** a scope sentence does not repair the general model's acceptance of sibling-domain requests on MASSIVE. The instruction raises Gemma's rejection by 12–20 points but costs 3–5 points of in-scope acceptance.

**Jev** (reported beside the general models, outside the families) follows the same pattern:
- same-source rows: +4.2 to +9.0;
- cross-source rows: −0.0 to +1.0;
- instruction: −0.1 to +0.9.

## What this changes

1. **MASSIVE's confirmatory ranking is not robust.** The specialist's lead was specific to holding out email and play. These are sibling domains of retained skills, which the general models map onto music and social. Across held-out domains generally, the specialist and Gemma are level, and Qwen is slightly ahead.
2. **Labelled out-of-scope examples help a prompted model only when they cover the domains that will actually be requested.** Examples of other unsupported domains barely transfer. That limits run C's +13 points as a deployment lever.
3. **The one-sentence instruction does not fix sibling-domain acceptance on MASSIVE.**
