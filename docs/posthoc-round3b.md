# Post hoc re-analyses requested by the round 3 review (2026-10-16)

**POST HOC.** These were requested by `docs/reviews/round3/synthesis.md` §6 (free work), and none was pre-registered.

- **Code:** `src/bakeoff/posthoc_round3b.py`.
- **Numbers:** `artifacts/posthoc/round3b.json` and `artifacts/posthoc/b77oos_label_audit.json`.

## A. BANKING77-OOS label audit

Rows were matched to `legacy-datasets/banking77` after lower-casing. 5,903 of the 5,905 training rows matched.

| What | Count |
|---|---|
| Training rows with the wrong label | 366 |
| Of those, held-out-intent utterances labelled with an in-scope code (**leaked**) | 145, from 17 of the 27 held-out intents |
| Run D in-scope test items with a wrong gold intent | 32 of 398 |
| Run D "held-out" test items whose original label is an in-scope intent | 36 of 679 |

The local files are byte-identical to the upstream release (`fa955331`), so the error is in the published dataset. The leaked row indices and their SHA-256 are in the audit file. CLINC single-domain agrees with `clinc_oos` throughout (reviewer 2).

## B. Run D and Jev J1 on leak-free subsets of BANKING77-OOS

The table shows held-out rejection, zero-shot → retrieval, with the drop (one-sided sign flip clustered by held-out intent). The leak-free subset keeps only:
- the 10 held-out intents that never leak;
- label-clean items: 366 in-scope and 240 held-out.

| Model | As run (27 intents) | Leak-free subset | Δd′ / Δc on the leak-free subset (post hoc signal detection) |
|---|---|---|---|
| Gemma 4 31B | 59.5 → 30.3 (−29.2) | 67.5 → 42.9 (−24.6, p = .004) | +0.17 / +0.72 |
| Qwen3.5 27B | 82.2 → 58.5 (−23.7) | 85.4 → 71.2 (−14.2, p = .008) | +0.43 / +0.71 |
| Llama 3.3 70B | 50.7 → 17.7 (−33.0) | 69.2 → 31.7 (−37.5, p = .009) | +0.26 / +1.11 |
| DeepSeek V4 Pro | 60.8 → 21.1 (−39.8) | 70.4 → 38.3 (−32.1, p = .004) | +0.44 / +1.05 |
| Jev 1.13.0 | 70.7 → 25.3 (−45.4) | 79.2 → 50.4 (−28.7, p = .004) | AUROC 0.915 → 0.947, Δ +0.03 [+0.007, +0.068]; mean P(in scope) on held-out requests +0.17 |

With 10 clusters, the smallest one-sided p-value the sign-flip test can produce is about .001.

**What this shows:**
- **The suppression survives the leak** for all five models, at 14–38 points.
- **The leak inflated it.** The as-run drops were 24–45 points, and the models copy leaked codes.
- **Mechanism, now stated more precisely.** On leak-free items, retrieval moves every model a long way toward acceptance (Δc +0.7 to +1.1; Jev's mean in-scope probability on held-out requests +0.17). Discrimination stays level or improves slightly (Δd′ +0.2 to +0.4; Jev's AUROC +0.03). The acceptance shift dominates, which is why rejection still falls.
- **Jev's pre-registered "operating-point shift" readout** (AUROC unchanged) holds on the as-run items. On leak-free items the AUROC rises a little, so the leak masked a small discrimination gain.
- **The "closeness" explanation is withdrawn.** With the leaked rows removed from the pool, the dense-similarity AUROC for held-out against in-scope items is 0.90 (0.94 on the leak-free subset), level with CLINC single-domain's 0.91. Why the two splits differ remains open; option count and prompt structure are the candidates for run D2.

## C. Mitigations on held-out intents not covered by dev

**Coverage of the dev items.**
- The dev out-of-scope items covered 5 of 27 held-out intents on BANKING77-OOS (122 test items) and 2 of 5 per CLINC single-domain domain (280 test items).
- The specialist rejects 64.8% of covered-intent items against 44.5% of uncovered ones on BANKING77-OOS, and 78.2% against 37.9% on CLINC single-domain.

**Mitigations on uncovered intents only** (rejected; share of the lost rejection recovered; in-scope-right change with 95% CI):

| Split | Rule | Gemma | Qwen 27B | Llama 70B | DeepSeek |
|---|---|---|---|---|---|
| BANKING77-OOS | retrieval + specialist veto | 52.6; 0.80; −2.3 [−3.6, −1.0] | 67.9; 0.42; −2.0 [−3.3, −0.9] | 47.2; 0.96; −3.0 [−4.9, −1.4] | 47.4; 0.68; −2.8 [−4.6, −1.3] |
| BANKING77-OOS | retrieval + dense floor | 67.0; 1.25; −14.8 | 76.5; 0.74; −14.3 | 61.9; 1.39; −15.3 | 61.6; 1.00; −15.8 |

**No rule meets run D's pre-registered "works" criterion** (recover at least half of the loss, with the in-scope-right change interval above −2 points), for any model on either split:
- The veto recovers 42–96% of the loss on BANKING77-OOS, but its in-scope cost interval reaches beyond −2 points every time.
- The floors cost 11–16 points of in-scope accuracy.

## D. Freshness: confirmatory comparisons on never-seen turns

A turn counts as "never seen" if its normalised text appears in no session built before the confirmatory run, in any casing.

| Set | Turns never seen | Of which out of scope |
|---|---|---|
| Banking77 | 582 of 1,479 | 119 |
| MASSIVE | 1,215 of 1,216 | 243 |
| CLINC150 | 1,766 of 1,850 | 107 |

Results on never-seen turns (Holm over the same six primary tests):

| Set | Specialist − Gemma | Holm | Combination − Gemma | Holm |
|---|---|---|---|---|
| Banking77 | −1.65 [−4.02, +0.68] | .94 | +0.98 [−0.42, +2.40] | .94 |
| MASSIVE | +4.42 [−0.04, +8.09] | .94 | +0.60 | .94 |
| CLINC150 | −0.94 [−2.14, +0.35] | .94 | +0.10 | .94 |

**On never-seen turns no primary comparison is significant.** The only significant pre-registered result (CLINC150, −1.89, Holm .007) does not survive restriction to never-seen turns.

**Secondary comparisons on never-seen turns:**

| Comparison | Banking77 | MASSIVE | CLINC150 |
|---|---|---|---|
| Gemma with out-of-scope rows − Gemma | +0.2 | +13.3 (p = .0001) | +0.8 (p = .064) |
| Jev with retrieval − Gemma | −0.9 (n.s.) | +4.0 (p = .0001) | −0.2 (n.s.) |
| Specialist − Qwen 27B | n.s. | n.s. | −1.7 (p = .018) |

## E. MASSIVE

**Label coverage.**
- The in-scope turns cover **21 of the 42 labels**. Five intents make up 62% of them: calendar set (175), general quirky (134), weather query (112), qa factoid (100) and calendar query (82).
- The dev set covers 24 labels and misses 9 of the 21 labels that appear in the test turns.

**Sensitivity check:** without `general quirky`, specialist − Gemma = **+6.1** [+2.4, +8.8], p = .001.

**Out-of-scope requests rejected, by held-out domain (%):**

| Domain | Turns | Specialist | Gemma | Qwen 27B | Gemma with rows | Jev retrieval | Kimi K3* | Claude Opus 5* |
|---|---|---|---|---|---|---|---|---|
| play | 130 | 78.5 | **5.4** | 54.6 | 85.4 | 22.3 | 9.8 | 7.8 |
| email | 98 | 80.6 | 34.7 | 81.6 | 95.9 | 71.4 | 54.1 | 54.1 |
| iot | 15 | 93.3 | 100 | 100 | 100 | 53.3 | 100 | 100 |

\* First 100 sessions only.

Every general model, including both frontier models, accepts nearly all *play* requests, mapping them onto the retained music skills. The held-out domains are sibling domains of kept skills, so this is a near-domain out-of-scope result.

## F. Minimum detectable differences and the equivalence bound

**Minimum detectable difference** at 80% power under the pre-registered weighted sign-flip test (2.8 × the SD of its null distribution):

| Set | MDD (points) |
|---|---|
| Banking77 | 1.9 |
| MASSIVE | **8.8** |
| CLINC150 | 1.7 |

**Banking77 equivalence bound.** Across 40 bootstrap seeds, the 90% lower bound averages −2.51 with SD 0.03, and it falls inside the ±2.5 margin for only 37.5% of seeds. The equivalence result is **indeterminate**, not "missed by 0.03".

## G. Cost on one basis, hosting break-even, host table, cross-host variance

**Cost per 1,000 sessions, uncached at list price, as a multiple of Gemma** (Kimi's billed cost included 44–66% cache hits):

| Model | Banking77 | MASSIVE | CLINC150 |
|---|---|---|---|
| Qwen 27B | 1.4× | 1.5× | 1.4× |
| Kimi K3 | 20.5× | 21.3× | 17.4× |
| Claude Opus 5 | 44.1× | 43.0× | 36.1× |

**Hosting break-even.** Hosting the specialist costs less than calling Gemma on every turn above the following volumes (sessions per month):

| Hosting cost | Banking77 | MASSIVE | CLINC150 |
|---|---|---|---|
| $50 per month | 68k | 106k | 47k |
| $255 per month | 347k | 543k | 240k |

**Hosts and precision** (prices as listed on 2026-10-15/16):

| Model | Host | Precision |
|---|---|---|
| Gemma 4 31B | Novita | bf16 |
| Qwen3.5 27B | Alibaba | not stated by host |
| Kimi K3 | Moonshot AI | mxfp4 |
| Claude Opus 5 | Anthropic | not stated by host |
| Llama 3.3 70B | DeepInfra | fp8 |
| DeepSeek V4 Pro | StreamLake | fp8 |
| Jev | TypeSafe | pinned 1.13.0; precision not stated |

**Cross-host variance.** DeepSeek V4 Pro was run on two hosts, Parasail and StreamLake, both fp8, with the same 686 prompts at temperature 0:
- 9.5% of replies differ;
- 5.2% of scope decisions flip;
- held-out rejection is 63.4% against 59.5%, a 3.9-point difference.

This is the size of several differences the paper reports.
