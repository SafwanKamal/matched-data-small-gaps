# Confirmatory run: results (2026-10-16)

Protocol: `confirmatory_protocol.json` (frozen before any request, with a dated frontier amendment; see `release/FREEZE_TABLE.md` for its freeze time, frozen text and item-list hash). Analysis: `src/bakeoff/confirmatory.py analyze` → `artifacts/confirmatory/analysis.json`. Figure: `docs/figures/f5_cost_frontier.{pdf,png}`.

Fresh sessions built from test items no earlier session used; text lower-cased with punctuation stripped; native out-of-scope data. Banking77 uses 184 CLINC out-of-scope queries. MASSIVE holds out the email, iot and play domains, leaving 42 labels. CLINC150 uses 185 native out-of-scope queries.

*Correction, 2026-10-17, round 5 review:* "no earlier session used" is wrong. The freshness rule missed the exploratory BANKING77-OOS sessions: 832 of 1,295 Banking77 in-scope turns, 65 of its 184 out-of-scope turns and 78 of CLINC150's 185 out-of-scope turns had appeared there; MASSIVE is unaffected (1,215 of 1,216 turns never seen). The run was executed and analysed as frozen. See the dated erratum `erratum_2026-10-16_freshness` in `confirmatory_protocol.json` and the never-seen sensitivity analysis in `docs/posthoc-round3b.md` (D).

- **Specialist:** Ettin 150M, retrained on the normalised text; 3 seeds averaged; prior-weighted threshold.
- **General models:** Gemma 4 31B (primary) and Qwen3.5 27B, each given 20 BM25 examples from in-scope training rows only.
- **Endpoint:** weighted turn accuracy, 0.8 × in-scope + 0.2 × out-of-scope.
- **Tests:** sign-flip clustered by intent; Holm over the 6 primary tests.
- **Failures:** every request of every run was answered and parsed (0 failures, 0 parse failures).

## Pre-registered outcomes (primary family, Holm over 6)

| Set | Specialist − Gemma (points, 95% CI) | p (Holm) | Combination − Gemma | p (Holm) |
|---|---|---|---|---|
| Banking77 | −1.42 [−2.69, −0.14] | .037 (.19) | +0.58 [−0.06, +1.21] | .099 (.40) |
| MASSIVE (42 skills) | +4.42 [−0.04, +8.10] | .17 (.51) | +0.60 [−0.15, +1.18] | .30 (.59) |
| CLINC150 | −1.89 [−2.99, −0.78] | .0012 (**.0072**) | +0.10 [−0.12, +0.35] | .52 (.59) |

The sentences the protocol fixed for these outcomes:

- **Banking77.** The difference (−1.42 [−2.69, −0.14]) is not significant after Holm, and equivalence within 2.5 points is not shown: the 90% interval [−2.53, −0.35] misses the margin by 0.03. We cannot rank the two.
- **MASSIVE.** The difference (+4.42 [−0.04, +8.10]) is not significant; we cannot rank the two.
- **CLINC150.** The retrieval-prompted general model beats the fine-tuned specialist by 1.89 points [0.78, 2.99]. The gap is split between scope (−0.88) and intent (−1.01).
- **Combination.** On all three sets the combination is level with the general model alone. The post hoc Banking77 gain (+1.0) shrinks to +0.58 and is not significant.
- **Headline rule.** On the only significant set (CLINC150), the scope component is less than half of the gap. So the abstract may **not** say "the difference lies in the out-of-scope decision"; it reports the components as found.

## Secondary results (unadjusted, labelled as such)

- **The out-of-scope rows in the general model's pool matter enormously on MASSIVE.** Without them, Gemma rejects only 23.0% of requests from the held-out domains: it maps email, iot and play requests onto nearby skills. With 250 labelled out-of-scope rows in the pool it rejects 90.5%, and weighted accuracy rises by 13.3 points [12.1, 14.6] (p = .0001). The rows also help on CLINC150 (+0.72, p = .016) but make no difference on Banking77 (+0.17, p = .50). With the rows on both sides, the specialist (kNN vote) trails Gemma on Banking77 (−1.59) and CLINC150 (−2.61), and the two are level on MASSIVE (−1.81, p = .54). This echoes the retrieval-suppression finding: in-scope examples make the general model accept nearby unsupported requests.
- **Qwen3.5 27B behaves differently on MASSIVE.** It rejects 68.3% of held-out-domain requests without the rows, and beats the specialist there by 2.57 points (p = .26, not significant). On Banking77 and CLINC150 it is level with Gemma.
- **The cascade (40% of turns to the model) matches the model alone at 40% of its cost.** Cascade minus model is +0.18, −0.36 and +0.08 points (none significant). Cascade minus specialist is +1.60, −4.78 and +1.97.
- **Threshold sensitivity (specialist − Gemma).** At a 10% out-of-scope rate: −1.78, +2.50, −1.89. At 30%: −1.25, +3.82, −1.63. The direction is the same as at 20%.
- **Normalising the text** changes the specialist little: original-case training and scoring minus normalised gives −0.18, −1.73 (p = .03) and +0.54. Normalisation helps slightly on MASSIVE. The residual length cue is weak on every set (AUROC 0.59, 0.56, 0.57).
- **Frontier models, first 100 sessions per set (500 turns each), same prompts, reasoning off:**

  | Set | Specialist | Gemma 4 31B | Kimi K3 (2.8T) | Claude Opus 5 |
  |---|---|---|---|---|
  | Banking77 | 93.0 | 95.0 | 95.9 | 95.9 |
  | MASSIVE | 80.4 | 78.5 | 79.0 | 77.2 |
  | CLINC150 | 93.2 | 95.9 | 95.1 | 97.6 |

  - Neither frontier model is significantly better than Gemma on any set. The largest gap is Opus on CLINC150, +1.78 (p = .065).
  - They beat the specialist on Banking77 (Kimi +2.9, Opus +2.9, both p < .01), and Opus does on CLINC150 (+4.4, p = .0001).
  - On MASSIVE, where held-out-domain rejection decides the score, no general model beats the specialist.
  - **The owner's question — "would trillion-parameter models crush this?" — is answered no.** At 8–40× Gemma's cost per session, Kimi K3 and Claude Opus 5 are within about 2 points of a 31B model given retrieved examples.

## Tokens, cost and latency per system

Rows marked * are the first 100 sessions only.

- **Costs** are measured: the provider-returned cost of every call.
- **Combination** calls the model on every turn.
- **Cascade** calls it on 40% of turns: its cost per session is scaled, while its tokens and latency are per model call.
- **Latency** is per call.
- **Gemma's Banking77 p95** (29 s) reflects a Novita slowdown during run A.
- **Specialist latency** is from `artifacts/program2/banking77-log_loss/metrics.json` (bf16, RTX 5070).


**banking77**

| System | Weighted acc. | Session success | Input / output tokens per turn | USD per 1,000 sessions | USD per correct session | Latency p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| Specialist (3 seeds) | 92.7 | 70.3 | n/a | 0 (local GPU) | 0 | 11 / 13 (RTX 5070) |
| Gemma 4 31B | 94.1 | 73.6 | 1022 / 10 | 0.73 | 0.0010 | 1014 / 28896 |
| Gemma + specialist (combination) | 94.7 | 76.2 | 1022 / 10 | 0.73 | 0.0010 | 1014 / 28896 |
| Gemma cascade 0.4 | 94.3 | 74.3 | 1022 / 10 | 0.29 | 0.0004 | 1014 / 28896 |
| Gemma, pool with out-of-scope rows | 94.3 | 74.0 | 1021 / 10 | 0.73 | 0.0010 | 1109 / 53535 |
| Qwen3.5 27B | 94.0 | 72.3 | 1015 / 9 | 1.06 | 0.0015 | 3642 / 12527 |
| Qwen + specialist (combination) | 94.5 | 75.3 | 1015 / 9 | 1.06 | 0.0014 | 3642 / 12527 |
| Qwen cascade 0.4 | 94.3 | 73.8 | 1015 / 9 | 0.42 | 0.0006 | 3642 / 12527 |
| Kimi K3 (2.8T) * | 95.9 | 81.0 | 914 / 18 | 8.25 | 0.0102 | 2828 / 6097 |
| Claude Opus 5 * | 95.9 | 81.0 | 1233 / 13 | 32.43 | 0.0400 | 1164 / 2565 |
| Gemma 4 31B, same 100 sessions * | 95.0 | 78.0 | 1016 / 10 | 0.73 | 0.0009 | 6211 / 57628 |

**massive**

| System | Weighted acc. | Session success | Input / output tokens per turn | USD per 1,000 sessions | USD per correct session | Latency p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| Specialist (3 seeds) | 78.3 | 28.0 | n/a | 0 (local GPU) | 0 | 11 / 13 (RTX 5070) |
| Gemma 4 31B | 73.8 | 24.2 | 644 / 10 | 0.47 | 0.0019 | 875 / 1791 |
| Gemma + specialist (combination) | 74.4 | 24.5 | 644 / 10 | 0.47 | 0.0019 | 875 / 1791 |
| Gemma cascade 0.4 | 73.5 | 23.9 | 644 / 10 | 0.19 | 0.0008 | 875 / 1791 |
| Gemma, pool with out-of-scope rows | 87.2 | 52.5 | 643 / 10 | 0.47 | 0.0009 | 832 / 1806 |
| Qwen3.5 27B | 80.8 | 38.1 | 642 / 9 | 0.69 | 0.0018 | 1330 / 10307 |
| Qwen + specialist (combination) | 81.7 | 39.1 | 642 / 9 | 0.69 | 0.0018 | 1330 / 10307 |
| Qwen cascade 0.4 | 80.4 | 37.0 | 642 / 9 | 0.28 | 0.0007 | 1330 / 10307 |
| Kimi K3 (2.8T) * | 79.0 | 31.0 | 578 / 18 | 6.59 | 0.0213 | 2958 / 5278 |
| Claude Opus 5 * | 77.2 | 36.0 | 746 / 13 | 20.20 | 0.0561 | 1520 / 1952 |
| Gemma 4 31B, same 100 sessions * | 78.5 | 35.0 | 640 / 10 | 0.47 | 0.0013 | 951 / 2088 |

**clinc150**

| System | Weighted acc. | Session success | Input / output tokens per turn | USD per 1,000 sessions | USD per correct session | Latency p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| Specialist (3 seeds) | 93.1 | 74.4 | n/a | 0 (local GPU) | 0 | 11 / 13 (RTX 5070) |
| Gemma 4 31B | 95.0 | 83.2 | 1485 / 11 | 1.06 | 0.0013 | 1006 / 2288 |
| Gemma + specialist (combination) | 95.1 | 83.8 | 1485 / 11 | 1.06 | 0.0013 | 1006 / 2288 |
| Gemma cascade 0.4 | 95.1 | 83.4 | 1485 / 11 | 0.42 | 0.0005 | 1006 / 2288 |
| Gemma, pool with out-of-scope rows | 95.7 | 84.1 | 1485 / 11 | 1.06 | 0.0013 | 894 / 1562 |
| Qwen3.5 27B | 95.0 | 81.4 | 1481 / 10 | 1.52 | 0.0019 | 2251 / 4829 |
| Qwen + specialist (combination) | 95.1 | 82.0 | 1481 / 10 | 1.52 | 0.0019 | 2251 / 4829 |
| Qwen cascade 0.4 | 95.1 | 82.1 | 1481 / 10 | 0.61 | 0.0007 | 2251 / 4829 |
| Kimi K3 (2.8T) * | 95.1 | 82.0 | 1142 / 18 | 8.28 | 0.0101 | 2810 / 4791 |
| Claude Opus 5 * | 97.6 | 88.0 | 1470 / 13 | 38.35 | 0.0436 | 1084 / 1582 |
| Gemma 4 31B, same 100 sessions * | 95.9 | 84.0 | 1488 / 11 | 1.06 | 0.0013 | 949 / 1954 |

## Spend

Runs A, B, C and I cost 13.93 USD (A 0.73, B 1.06, C 0.72, I 11.42: Kimi K3 about 2.3, Claude Opus 5 about 9.1). The plumbing checks and the frontier probe cost 0.30 USD before them. Project total: 60.12 USD of the 80 USD cap. The total was rebuilt from saved rows because two processes briefly ran in parallel (see the protocol's execution note).
