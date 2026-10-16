"""Decoder continuation rules. No GPU and no dataset rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bakeoff.train_ettin import decision_loss  # noqa: E402
from bakeoff.train_ettin_decoder import (  # noqa: E402
    _continuation_ids,
    _loss_grad,
    _mean_continuation_logprob,
)


class _FakeTokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


def test_label_is_a_strict_continuation() -> None:
    spec = {
        "prompt_suffix": "\n",
        "prepend_bos": True,
        "max_prompt_tokens": 128,
    }
    ids, prompt_len, truncated = _continuation_ids(_FakeTokenizer(), "hi", "positive", spec)
    assert truncated is False
    assert ids[0] == 1
    assert prompt_len == 1 + len("hi\n")
    assert ids[prompt_len:] == [ord(char) for char in "positive"]


def test_mean_logprob_of_known_tokens() -> None:
    # Sequence [bos, a, b]. Score only b. Logits at position 1 predict b.
    logits = torch.zeros(1, 3, 4)
    logits[0, 1, 3] = 10.0
    input_ids = torch.tensor([[1, 2, 3]])
    prompt_lens = torch.tensor([2])
    lengths = torch.tensor([3])
    score = _mean_continuation_logprob(logits, input_ids, prompt_lens, lengths)
    expected = torch.log_softmax(logits[0, 1].float(), dim=-1)[3]
    assert torch.allclose(score, expected, atol=1e-5)


def test_loss_grad_matches_autograd() -> None:
    gold = torch.tensor([0, 2, 4, 1])
    for kind in ("log_loss", "brier"):
        scores = torch.randn(4, 5)
        leaf = scores.detach().clone().requires_grad_(True)
        decision_loss(leaf, gold, kind).backward()
        analytic = _loss_grad(scores, gold, kind)
        assert torch.allclose(leaf.grad, analytic, atol=1e-5)


def test_brier_matches_hand_count() -> None:
    scores = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    loss = decision_loss(scores, torch.tensor([1]), "brier")
    assert torch.allclose(loss, torch.tensor(0.5))
