"""Three Choice adapters. Each returns one pre-softmax score per frozen label."""

from __future__ import annotations

import gc

import numpy as np
import torch

from bakeoff.metrics import softmax


def _empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_parameters_on_cuda(module: torch.nn.Module, name: str) -> str:
    devices = sorted({parameter.device.type for parameter in module.parameters()})
    if devices != ["cuda"]:
        raise RuntimeError(f"{name} parameters are on {devices}, not cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.cuda.get_device_name(0)


class DecisionSystem:
    name = "base"

    def load(self) -> None:
        raise NotImplementedError

    def score(self, text: str, labels: list[str]) -> tuple[np.ndarray, bool]:
        """Return pre-softmax scores and whether the packed input was truncated."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def device_name(self) -> str:
        raise NotImplementedError


class DebertaChoice(DecisionSystem):
    name = "deberta-v3-large-zeroshot-v2.0-c"

    def __init__(self, spec: dict):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.entailment_id = None
        self.dtype = torch.bfloat16
        self.dtype_name = "bfloat16"
        self.dtype_fallback = False

    def load(self, dtype: torch.dtype | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.close()
        self.dtype = dtype or torch.bfloat16
        self.dtype_name = "bfloat16" if self.dtype == torch.bfloat16 else "float32"
        revision = self.spec["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["hf_id"], revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.spec["hf_id"], revision=revision, dtype=self.dtype
        ).to("cuda")
        self.model.eval()
        entailment_id = None
        for label, index in self.model.config.label2id.items():
            if str(label).lower().startswith("entail"):
                entailment_id = int(index)
                break
        if entailment_id is None:
            raise RuntimeError("DeBERTa config has no entailment label")
        self.entailment_id = entailment_id
        assert_parameters_on_cuda(self.model, self.name)

    def fallback_to_float32(self) -> None:
        self.dtype_fallback = True
        self.load(dtype=torch.float32)

    def device_name(self) -> str:
        return assert_parameters_on_cuda(self.model, self.name)

    def score(self, text: str, labels: list[str]) -> tuple[np.ndarray, bool]:
        template = self.spec["hypothesis_template"]
        hypotheses = [template.format(label) for label in labels]
        untruncated = self.tokenizer(
            [text] * len(labels),
            hypotheses,
            truncation=False,
            padding=False,
        )
        truncated = any(len(ids) > self.spec["max_length"] for ids in untruncated["input_ids"])
        encoded = self.tokenizer(
            [text] * len(labels),
            hypotheses,
            truncation="only_first",
            max_length=self.spec["max_length"],
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        scores = logits[:, self.entailment_id].float().detach().cpu().numpy().astype(np.float64)
        if scores.shape != (len(labels),):
            raise RuntimeError(f"DeBERTa returned shape {scores.shape}")
        return scores, truncated

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        _empty_cuda()


class GLiClassChoice(DecisionSystem):
    name = "gliclass-base-v3.0"

    def __init__(self, spec: dict, device: str = "cuda"):
        self.spec = spec
        # The checkpoint is float32 on either device. A cpu run is for scores only, never for latency.
        self.device = device
        self.model = None
        self.tokenizer = None
        self.inner = None
        self.dtype_name = "float32"
        self.dtype_fallback = False

    def load(self) -> None:
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        self.close()
        revision = self.spec["revision"]
        self.model = GLiClassModel.from_pretrained(self.spec["hf_id"], revision=revision).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["hf_id"], revision=revision)
        pipe = ZeroShotClassificationPipeline(
            self.model,
            self.tokenizer,
            classification_type="single-label",
            device="cuda:0" if self.device == "cuda" else "cpu",
            max_length=self.spec["max_length"],
            max_classes=self.spec["max_labels_per_forward"],
            progress_bar=False,
        )
        self.inner = pipe.pipe
        if self.inner.classification_type != "single-label":
            raise RuntimeError("GLiClass pipeline is not single-label")
        if self.device == "cuda":
            assert_parameters_on_cuda(self.model, self.name)
        if next(self.model.parameters()).dtype != torch.float32:
            raise RuntimeError("GLiClass checkpoint was expected in float32")

    def device_name(self) -> str:
        return assert_parameters_on_cuda(self.model, self.name)

    def _chunk_logits(self, text: str, labels: list[str]) -> tuple[torch.Tensor, bool]:
        width = self.spec["max_labels_per_forward"]
        pieces = []
        truncated = False
        for start in range(0, len(labels), width):
            chunk = labels[start : start + width]
            packed = self.inner.prepare_input(text, chunk)
            length = len(self.tokenizer(packed, truncation=False, add_special_tokens=True)["input_ids"])
            truncated = truncated or length > self.spec["max_length"]
            encoded = self.inner.prepare_inputs([text], chunk, same_labels=True)
            with torch.inference_mode():
                output = self.inner.model(**encoded, max_num_classes=len(chunk))
            logits = output.logits[0, : len(chunk)].float()
            if logits.shape != (len(chunk),):
                raise RuntimeError(f"GLiClass chunk logits shape {tuple(logits.shape)}")
            pieces.append(logits)
        return torch.cat(pieces), truncated

    def score(self, text: str, labels: list[str]) -> tuple[np.ndarray, bool]:
        logits, truncated = self._chunk_logits(text, labels)
        if logits.shape != (len(labels),):
            raise RuntimeError(f"GLiClass logits shape {tuple(logits.shape)}")
        # The delivered simplex is softmax over the full label set, including
        # logits that were computed in separate forwards when the set exceeds 25.
        scores = logits.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(softmax(scores)).all():
            raise RuntimeError("GLiClass softmax was non-finite")
        return scores, truncated

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.inner = None
        _empty_cuda()


class QwenChoice(DecisionSystem):
    name = "qwen3-0.6b"

    def __init__(self, spec: dict):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.dtype = torch.bfloat16
        self.dtype_name = "bfloat16"
        self.dtype_fallback = False

    def load(self, dtype: torch.dtype | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.close()
        self.dtype = dtype or torch.bfloat16
        self.dtype_name = "bfloat16" if self.dtype == torch.bfloat16 else "float32"
        revision = self.spec["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["hf_id"], revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec["hf_id"], revision=revision, dtype=self.dtype
        ).to("cuda")
        self.model.eval()
        assert_parameters_on_cuda(self.model, self.name)

    def fallback_to_float32(self) -> None:
        self.dtype_fallback = True
        self.load(dtype=torch.float32)

    def device_name(self) -> str:
        return assert_parameters_on_cuda(self.model, self.name)

    def _prompt(self, text: str, labels: list[str]) -> str:
        block = "\n".join(f"- {label}" for label in labels)
        content = (
            "Classify the text into exactly one label. Do not explain.\n\n"
            f"Labels:\n{block}\n\n"
            f"Text:\n{text}\n"
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        suffix = self.spec["nonthinking_suffix"]
        if not prompt.endswith(suffix):
            raise RuntimeError("Qwen prompt is missing the non-thinking suffix")
        return prompt

    def score(self, text: str, labels: list[str]) -> tuple[np.ndarray, bool]:
        prompt = self._prompt(text, labels)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        truncated = len(prompt_ids) > self.spec["context_limit"]
        sequences = []
        for label in labels:
            full = self.tokenizer.encode(prompt + label, add_special_tokens=False)
            if full[: len(prompt_ids)] != prompt_ids or len(full) == len(prompt_ids):
                raise RuntimeError(f"label is not a continuation of the prompt: {label}")
            sequences.append(full)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        scores = np.empty(len(labels), dtype=np.float64)
        prompt_len = len(prompt_ids)
        batch_size = int(self.spec["label_batch_size"])
        body = self.model.base_model
        head = self.model.get_output_embeddings()
        # Vocabulary logits are computed only at the positions that predict a label token.
        # Projecting every position of a prompt that lists all 77 Banking77 labels costs
        # about 600 MB per chunk and ran out of memory on the 12 GB card. Chunks stay
        # inside one decision.
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            width = max(len(sequence) for sequence in chunk)
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for row, sequence in enumerate(chunk):
                input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention[row, : len(sequence)] = 1
            with torch.inference_mode():
                hidden = body(
                    input_ids=input_ids.to("cuda"),
                    attention_mask=attention.to("cuda"),
                ).last_hidden_state
                for row, sequence in enumerate(chunk):
                    # Position p - 1 predicts token p.
                    steps = head(hidden[row, prompt_len - 1 : len(sequence) - 1]).float()
                    targets = torch.tensor(sequence[prompt_len:], dtype=torch.long, device=steps.device)
                    token_log_probs = steps.gather(-1, targets[:, None])[:, 0] - torch.logsumexp(steps, dim=-1)
                    scores[start + row] = float(token_log_probs.mean().cpu())
            del hidden
        if scores.shape != (len(labels),) or not np.isfinite(scores).all():
            raise RuntimeError("Qwen scores were the wrong shape or non-finite")
        return scores, truncated

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        _empty_cuda()


def build_systems(protocol: dict) -> list[DecisionSystem]:
    specs = protocol["systems"]
    return [
        DebertaChoice(specs["deberta-v3-large-zeroshot-v2.0-c"]),
        GLiClassChoice(specs["gliclass-base-v3.0"]),
        QwenChoice(specs["qwen3-0.6b"]),
    ]
