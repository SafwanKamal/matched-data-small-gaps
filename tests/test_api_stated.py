import math

import numpy as np

from bakeoff.api_extra import stated_distribution


def test_remainder_goes_to_unlisted_options():
    probs, listed, total = stated_distribution('{"2": 0.7, "1": 0.1}', 4)
    assert listed == 2 and math.isclose(total, 0.8)
    assert np.allclose(probs, [0.1, 0.7, 0.1, 0.1], atol=1e-6)


def test_overfull_reply_is_normalised():
    probs, _, total = stated_distribution('Here: {"01": 0.9, "12": 0.9}', 12)
    assert math.isclose(total, 1.8)
    assert math.isclose(probs[0], 0.5, abs_tol=1e-6) and math.isclose(probs[11], 0.5, abs_tol=1e-6)


def test_codes_are_padded_and_bad_entries_dropped():
    probs, listed, _ = stated_distribution('{"7": 1.0, "99": 0.5, "x": 0.2, "3": "high"}', 12)
    assert listed == 1
    assert math.isclose(probs[6], 1.0, abs_tol=1e-6)


def test_unusable_replies():
    assert stated_distribution("I think option 3.", 5)[0] is None
    assert stated_distribution('{"9": 0.5}', 5)[0] is None
    assert stated_distribution("{not json}", 5)[0] is None
