import math

import numpy as np

from bakeoff.api_baseline import codes_for, tree_distribution


def _step(token, p, alternatives):
    return {"token": token, "logprob": math.log(p), "top_logprobs": [{"token": name, "logprob": math.log(value)} for name, value in alternatives.items()]}


def test_single_digit_codes_follow_the_alternatives():
    content = [_step("2", 0.7, {"2": 0.7, "1": 0.2, "3": 0.1})]
    probs = tree_distribution(content, 3)
    assert np.allclose(probs, [0.2, 0.7, 0.1], atol=1e-6)


def test_uncovered_mass_is_split_among_unseen_options():
    content = [_step("1", 0.6, {"1": 0.6, "2": 0.2})]
    probs = tree_distribution(content, 4)
    assert np.allclose(probs, [0.6, 0.2, 0.1, 0.1], atol=1e-6)


def test_two_token_codes_use_the_generated_path():
    assert codes_for(12)[:2] == ["01", "02"]
    content = [
        _step("0", 0.9, {"0": 0.9, "1": 0.1}),
        _step("7", 0.5, {"7": 0.5, "3": 0.5}),
    ]
    probs = tree_distribution(content, 12)
    assert math.isclose(probs[6], 0.45, abs_tol=1e-6)
    assert math.isclose(probs[2], 0.45, abs_tol=1e-6)
    # Codes 10, 11 and 12 share the mass of the alternative first token.
    assert np.allclose(probs[9:12], 0.1 / 3, atol=1e-6)
    assert math.isclose(probs.sum(), 1.0, abs_tol=1e-9)


def test_whole_code_tokens_and_whitespace():
    content = [_step(" ", 1.0, {}), _step("07", 0.8, {"07": 0.8, "12": 0.2})]
    probs = tree_distribution(content, 12)
    assert math.isclose(probs[6], 0.8, abs_tol=1e-6)
    assert math.isclose(probs[11], 0.2, abs_tol=1e-6)


def test_text_that_matches_no_code_is_unscorable():
    assert tree_distribution([_step("The", 0.9, {"The": 0.9})], 5) is None
    assert tree_distribution(None, 5) is None
