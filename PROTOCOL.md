# Frozen protocol: Program 1 Choice bakeoff

Normative details that are lists, revisions, or counts live in `protocol.json`. This note is the same freeze in sentences. It was written before any test metric from this run.

The contract is Choice. Each system returns one probability per frozen label. Those probabilities are a simplex. The decision is the argmax, and a tie goes to the lowest label index. Nothing in the contract is a rationale. Output tokens are zero.

Temperature is one positive scalar per system and dataset. It is fit by minimizing mean NLL on the dev split, inside the bounds in `protocol.json`. Test metrics are computed after that fit. The primary table uses the scaled simplex. The unscaled simplex is reported beside it and is not used to choose the temperature.

## Data

SST-2 test is the entire GLUE validation split (872 labeled sentences). The GLUE test split has no gold labels and is not used. SST-2 dev is 400 rows from the GLUE train split.

AG News test is 1000 rows from the official test split (7600). AG News dev is 400 rows from the official train split. The state is the dataset `text` field. Labels, in index order, are World, Sports, Business, Sci/Tech.

Banking77 test is 500 rows from the official test split (3080). Banking77 dev is 400 rows from the official train split (10003). The loader is `legacy-datasets/banking77` because `datasets` 5.1.0 rejects the `PolyAI/banking77` script. The class names are the published names with each underscore replaced by a space. That replacement is the whole label edit.

Dev and test come from different official splits. The subsample is the prefix of a permutation from `numpy.random.SeedSequence(20261007).spawn`, in the order listed in `protocol.json`. SST-2 test is not permuted. Exact string overlap between dev and test is counted and does not drop a row.

## Systems

Label strings are shared.

DeBERTa (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c`) is the Hugging Face zero-shot path with `multi_label` false. The hypothesis is `This example is {label}.` The score is the entailment logit, identified as the `label2id` entry whose name starts with `entail`. The simplex is a softmax over those logits. Pairs are truncated with `only_first` at 512 tokens. The forward is bfloat16.

GLiClass (`knowledgator/gliclass-base-v3.0`) is `gliclass` 0.1.20 single-label mode: softmax over the label logits, not independent sigmoids. There is no task prompt and no few-shot example. Inputs are packed by `UniEncoderZeroShotClassificationPipeline.prepare_input`. The checkpoint scores at most 25 labels in one forward. Larger sets are scored in contiguous chunks of 25, in frozen label order; the logits are concatenated; one softmax is applied to the full vector. The checkpoint dtype is float32. Sequence length is capped at 512, the backbone maximum.

Qwen (`Qwen/Qwen3-0.6B`) is a closed-set readout. The user message lists the frozen labels and the text and says to classify without an explanation. `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` builds the prompt. On this tokenizer that appends an empty `<think>` block. That block is template text, not a sampled rationale. `generate` is not called. Each label is appended to the prompt with no extra space, and the score is the mean log-probability of the continuation tokens. The simplex is a softmax over those means. The forward is bfloat16. Label continuations are forwarded in batches of 4 inside the one timed decision. A single forward that materializes the full vocabulary logits for all 77 Banking77 labels does not fit in 12 GB. That batch size was fixed after a synthetic-sentence allocation failure, before any test metric.

A packed input that exceeds the stated length is still scored and counted as truncated. It is not dropped.

## Metrics

Accuracy, multiclass Brier (mean over items of the sum of squared errors across classes), and natural-log loss with the gold probability clipped at 1e-12.

Top-label ECE uses 15 equal-width bins on [0, 1]. Bin `i < 14` is `[i/15, (i+1)/15)`. The last bin is `[14/15, 1]`. Empty bins are omitted. The weight of a bin is its count over the number of scored items.

Selective coverage ranks items by top-label probability, highest first, with ties left in their original order. Coverage at 5% error is the largest prefix whose error rate is at most 0.05. If no prefix qualifies, coverage is 0. AURC is the trapezoid of that risk curve against coverage `k/n`.

Schema validity is the fraction of test items whose delivered distribution has one finite probability per frozen label, no probability below `-1e-6`, and a sum within `1e-4` of 1. The reported choice has to be one of those labels. Validity is not accuracy.

Latency is one text per timed call on the named GPU. Every label for that text is scored inside the call. `torch.cuda.synchronize` brackets the call. Twenty dev items are warmed up and are not part of the percentile. p50 and p95 use NumPy's linear percentile. Dev scoring used to fit temperature is not the latency sample.

## What this run does not do

It does not train, does not run a cascade, does not score a rationale, and does not change label wording after the test numbers exist.
