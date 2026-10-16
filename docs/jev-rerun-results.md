# Jev re-run: results (2026-10-16)

## Sources

- **Protocol:** `jev_rerun_protocol.json`, frozen at `7221619` before any Jev request.
- **Code:** `src/bakeoff/jev_rerun.py`.
- **Numbers:** `artifacts/jev-rerun/analysis.json`. Replies are in `artifacts/jev-rerun/J1/`, `artifacts/jev-rerun/J2/` and `artifacts/jev-rerun/check/`.

## Requests

- **Count:** 17,421 requests plus 150 check requests. Every one was answered.
- **Model:** every reply came from the pinned `jev-1.13.0`, with 0 failures.
- **Decision rule:** scope is "in" when the scope probability is at least 0.5, untuned.
- **Cost:** Jev reports tokens, not dollars. That is about 21M input tokens and, judging by the check rows, roughly 13M output tokens. At the earlier rate of about $0.00003 per decision this is roughly $0.5–2.5. The TypeSafe dashboard has the exact charge.

## J1: retrieval suppression (run D items)

| Split | Arm | Held-out rejected | In-scope right | AUROC (scope probability) | Mean P(in scope), in-scope / held-out |
|---|---|---|---|---|---|
| BANKING77-OOS | zero-shot | 70.7 | 71.9 | 0.888 | 0.814 / 0.378 |
| | retrieval | 25.3 | 87.9 | 0.891 | 0.949 / 0.667 |
| | instruction | 25.5 | 88.7 | 0.885 | 0.945 / 0.665 |
| CLINC single-domain | zero-shot | 55.4 | 90.1 | 0.873 | 0.867 / 0.484 |
| | retrieval | 56.9 | 97.9 | 0.980 | 0.960 / 0.437 |
| | instruction | 61.7 | 97.9 | 0.980 | 0.956 / 0.416 |

### HJ1 (pre-registered; one-sided, clustered by held-out intent, Holm over 2)

- **BANKING77-OOS:** retrieval lowers Jev's rejection of held-out same-domain requests by **45.4 points** [35.6, 55.3], Holm p = .0002. That is a larger drop than for any of the four general models (24–40).
- **CLINC single-domain:** no drop: −1.4 [−14.3, +12.3], p = .59.

So Jev splits the same way as the general models.

### HJ2 (pre-registered mechanism; intervals only)

The protocol's readout rule decided each split in advance from the AUROC change and the change in mean scope probability.

| Split | AUROC change (95% CI) | Mean P(in scope) change, held-out (95% CI) | Mean P(in scope) change, in-scope (95% CI) | Readout |
|---|---|---|---|---|
| BANKING77-OOS | +0.003 [−0.032, +0.040] | +0.29 [+0.23, +0.35] | +0.13 [+0.10, +0.17] | **operating-point shift** |
| CLINC single-domain | +0.107 [+0.035, +0.204] | −0.05 [−0.16, +0.06] | +0.09 [+0.05, +0.15] | **discrimination change** |

Measured on Jev's probabilities, this confirms the account that was only post hoc for the general models (`docs/posthoc-round3.md`):

- **Where unsupported requests resemble supported ones,** retrieval raises the model's readiness to accept without improving discrimination.
- **Where they don't,** retrieval improves discrimination instead.

### Instruction arm (secondary)

- **BANKING77-OOS:** no effect (+0.1 points, p = .50).
- **CLINC single-domain:** +4.9 points [1.4, 8.4], p = .015, at no in-scope cost.

## J2: confirmatory sessions (secondary, unadjusted)

Weighted turn accuracy, %:

| Set | Specialist | Gemma 4 31B | Jev zero-shot | Jev retrieval | Jev gate (retrieval) + specialist intent |
|---|---|---|---|---|---|
| Banking77 | 92.7 | 94.1 | 79.2 | 94.1 | 93.8 |
| MASSIVE (42) | 78.3 | 73.8 | 63.4 | 77.8 | 77.5 |
| CLINC150 | 93.1 | 95.0 | 87.3 | 95.0 | 94.0 |

Jev with retrieval against the other systems (paired, weighted sign-flip, clustered):

| Set | vs Gemma | vs specialist |
|---|---|---|
| Banking77 | −0.03 (p = .96) | +1.39 [+0.22, +2.63] (p = .024) |
| MASSIVE | **+3.95** [+1.79, +6.34] (p = .0001) | −0.47 (p = .96) |
| CLINC150 | −0.02 (p = .99) | +1.87 [+0.63, +3.02] (p = .003) |

- **Jev with retrieval matches Gemma 4 31B with retrieval** on Banking77 and CLINC150, and beats it on MASSIVE.
- **On MASSIVE** Jev rejects 44% of held-out-domain requests, against Gemma's 23%. It is still well below the specialist's 81%.
- **Scope-probability AUROC:**

  | Set | AUROC |
  |---|---|
  | Banking77 | 0.988 |
  | MASSIVE | 0.912 |
  | CLINC150 | 0.987 |

  These are threshold-free, so Jev's scope probability is a strong detector even where its 0.5 cut-off is too accepting.
- **Retrieval makes a large difference for Jev:** zero-shot it is 11–15 points behind its retrieval arm on every set.

## What this changes

1. **The acceptance-shift finding now has a pre-registered mechanism test.** A fifth model, of a different kind (a typed-decision model returning probabilities), shows the same split-dependent effect, and its probabilities measure discrimination directly without signal-detection assumptions. This answers reviewer 1's round 3 objection that "d′ unchanged" depends on an equal-variance assumption.
2. **A typed-decision model given the same retrieved examples is level with a 31B general model** on Banking77 and CLINC150. The 90% intervals of Jev − Gemma are [−0.88, +0.80] and [−0.73, +0.71] points. Jev is ahead on the confirmatory MASSIVE draw (+4.0), but not on the three robustness draws (+0.5, −0.8, −3.5). Jev therefore belongs in the paper as a system: it answers the paper's question of who should make the decision. The earlier cut rested on zero-shot runs on confounded sessions.

   *Correction, 2026-10-16, round 4 review:* an earlier version claimed Jev "sits lower on the cost axis, by roughly 3–5× per call on token counts". That is unsupported. Jev's API reports no price, and at Gemma's list price its token counts would cost 3.2–3.8× **more** per call. Jev's cost is unknown until the TypeSafe invoice is read.
