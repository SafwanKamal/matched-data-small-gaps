# Post hoc analyses of run D (2026-10-16)

**POST HOC.** These analyses were written after run D's results were seen, and none of them was pre-registered.

- **Code:** `src/bakeoff/posthoc_round3.py`.
- **Numbers:** `artifacts/posthoc/round3.json`.
- **Cost:** none. They use saved answers, local embeddings and the saved specialists.

They answer the three questions in `docs/reviews/round3/packet.md`, section 6.

## Q1. Is the suppression a specific blind spot or a shift in operating point?

Each model's scope decision is treated as a signal-detection task:

- **"Hit":** rejecting a held-out-intent request.
- **"False alarm":** rejecting an in-scope request.
- **d′** measures how well the model *separates* the two kinds of request. **c** measures how ready it is to *accept*; higher c means more accepting.
- **Intervals:** cluster bootstrap, resampling held-out intents and in-scope intents separately.

Change from zero-shot to retrieval:

| Split | Model | Δd′ (95% CI) | Δc (95% CI) |
|---|---|---|---|
| BANKING77-OOS | Gemma 4 31B | −0.03 [−0.41, +0.39] | **+0.74** [+0.57, +0.96] |
| | Qwen3.5 27B | +0.08 [−0.26, +0.43] | **+0.75** [+0.58, +0.93] |
| | Llama 3.3 70B | +0.25 [−0.27, +0.88] | **+1.07** [+0.84, +1.41] |
| | DeepSeek V4 Pro | +0.02 [−0.47, +0.63] | **+1.09** [+0.88, +1.41] |
| CLINC single-domain | Gemma 4 31B | **+0.47** [+0.07, +0.98] | +0.11 [−0.08, +0.33] |
| | Qwen3.5 27B | **+0.82** [+0.22, +1.36] | **+0.36** [+0.09, +0.66] |
| | Llama 3.3 70B | **+0.79** [+0.13, +1.49] | +0.26 [−0.08, +0.58] |
| | DeepSeek V4 Pro | **+0.83** [+0.17, +1.83] | **+0.38** [+0.07, +0.87] |

**Answer: an operating-point shift.**

- **BANKING77-OOS.** For all four models, retrieval leaves discrimination unchanged; every Δd′ interval contains 0. It moves the criterion toward acceptance by 0.7–1.1 standard deviations.
- **CLINC single-domain.** Retrieval shifts the criterion less (+0.1 to +0.4; the interval excludes 0 for 2 of 4 models). It also *improves* discrimination (+0.5 to +0.8, every interval above 0). The two effects cancel in the rejection rate, which is why H1 found no suppression there.
- **The general pattern:** retrieval always makes the model more accepting. Rejection of unsupported requests falls only where retrieval does not also make the model better at telling them apart.

**Contrast with off-topic requests** (the old 400 BANKING77-OOS sessions; exploratory data):

| Model | Off-topic rejected, zero-shot → retrieval | In-scope kept, zero-shot → retrieval |
|---|---|---|
| Gemma 4 31B | 99.2 → 99.0 | 90.0 → 98.3 |
| Qwen3.5 27B | 99.5 → 99.0 | 80.3 → 95.1 |

- The acceptance shift is present here too: in-scope kept rises.
- Off-topic requests are far enough from the examples that rejection stays at ceiling, and d′ rises (3.7 → 4.4 and 3.4 → 4.0).

## Q2. Why do the two splits disagree?

**1. The held-out requests are much closer to the supported ones on BANKING77-OOS.** Closeness is measured by the top-1 cosine similarity to the retrieval pool, using all-mpnet-base-v2:

| Split | In-scope | Held-out | AUROC, in-scope vs held-out |
|---|---|---|---|
| BANKING77-OOS | 0.888 | 0.774 | 0.77 |
| CLINC single-domain | 0.832 | 0.629 | 0.91 |

With BM25 scores the AUROCs are 0.72 and 0.76.

- On CLINC single-domain, the retrieved examples look visibly unlike a held-out request. That gives the model information it can use, so discrimination rises.
- On BANKING77-OOS the retrieved examples look as relevant for a held-out request as for a supported one.

**2. Zero-shot over-rejects more on BANKING77-OOS.** In-scope kept zero-shot:

| Split | In-scope kept, zero-shot |
|---|---|
| BANKING77-OOS | 81–92% |
| CLINC single-domain | 88–97% |

So there was more rejection for retrieval to remove.

**3. Label copying is partial.** We measured how often an accepted request's predicted intent is the most common label among its 20 retrieved examples:

| Split | Accepted held-out requests | Accepted in-scope requests |
|---|---|---|
| BANKING77-OOS | 42–55% | 84–86% |
| CLINC single-domain | 53–69% | 93–94% |

Accepted held-out requests are often given a label that is not the majority one. The model is not simply copying its neighbours.

**4. Per held-out intent, the size of the drop does not track closeness consistently.** The Spearman ρ between per-intent suppression and closeness runs from −0.03 to 0.60, and its sign is not consistent across models. With 27 and 10 intents this analysis has little power; we report it as inconclusive.

**5. The number of options** differs (50 vs 10) and could not be varied separately. It remains a confound.

## Q3. Zero-cost mitigations on the saved answers (reviewer 4's suggestions)

Five rules were applied to saved answers:

- **Neighbour-label rule:** reject when the predicted intent is not among the retrieved labels.
- **BM25 floor and dense floor:** reject when the top-1 similarity is below a threshold, chosen on dev for balanced accuracy.
- **Specialist veto:** accept only when the specialist (majority of three seeds) also accepts.
- **Instruction plus veto:** the instruction arm, with the specialist veto applied.

Both the floors and the veto use 100 held-out-intent dev items, so they are not available to a deployment that has no examples of unsupported intents. The instruction is the only lever that needs no such examples.

### BANKING77-OOS

Each cell gives the share of held-out requests rejected / the share of in-scope requests right.

| Rule | Gemma 4 31B | Qwen3.5 27B | Llama 3.3 70B | DeepSeek V4 Pro |
|---|---|---|---|---|
| Zero-shot | 59.5 / 75.9 | 82.2 / 66.3 | 50.7 / 60.8 | 60.8 / 71.6 |
| Retrieval | 30.3 / 86.7 | 58.5 / 85.9 | 17.7 / 87.7 | 21.1 / 88.4 |
| Instruction | 33.3 / 86.4 | 63.5 / 85.2 | 31.7 / 85.9 | 31.8 / 88.2 |
| Retrieval + neighbour-label rule | 36.4 / 86.7 | 58.9 / 85.9 | 21.9 / 87.7 | 24.2 / 88.4 |
| Retrieval + dense floor | 69.8 / 71.9 | 77.8 / 71.6 | 65.7 / 72.4 | 65.4 / 72.6 |
| Retrieval + specialist veto | 56.6 / 84.4 | 70.4 / 83.9 | 51.1 / 84.7 | 51.5 / 85.7 |

### CLINC single-domain

Each cell gives the balanced accuracy, the mean of held-out rejected and in-scope kept.

| Rule | Gemma 4 31B | Qwen3.5 27B | Llama 3.3 70B | DeepSeek V4 Pro |
|---|---|---|---|---|
| Retrieval | 73.2 | 83.2 | 69.2 | 65.6 |
| Instruction | 84.2 | 90.6 | 82.2 | 73.5 |
| Retrieval + dense floor | 86.6 | 88.8 | 84.6 | 85.0 |
| Instruction + specialist veto | 87.3 | 91.5 | 87.2 | 84.7 |

### Reading

- **The neighbour-label rule does almost nothing:** the model nearly always picks a label from its neighbours.
- **The similarity floors** recover rejection but cost 11–16 points of in-scope accuracy.
- **The specialist veto** on BANKING77-OOS recovers 26–34 points of held-out rejection for about 2–3 points of in-scope accuracy. It restores roughly the zero-shot rejection rate while keeping retrieval's intent accuracy.
- **On CLINC single-domain,** the instruction alone is most of the gain, and the veto adds a little more.

## What this changes in the paper's claim (Q3 of the packet)

Restated contribution, worded as post hoc where it rests on these analyses:

> Retrieving in-scope examples moves every general model we tested toward accepting requests. Where unsupported requests resemble the supported ones (BANKING77-OOS), retrieval does not improve the model's ability to tell them apart, so rejection of unsupported same-domain requests falls by 24–40 points (pre-registered, 4 of 4 models). Where they are less alike (CLINC single-domain), retrieval also improves discrimination and rejection does not fall (pre-registered, 0 of 4). An explicit scope instruction lowers the acceptance shift, and requiring the fine-tuned classifier to agree recovers most of the lost rejection at a small in-scope cost (post hoc).
