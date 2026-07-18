"""Learning router (intelligent-model-routing R8, task 11.3).

Pins Property 7: record/bias per (category, model), neutral with no history,
bounded store, and persistence round-trip via a prefs blob.
"""
from __future__ import annotations

from app.llm import learning


def setup_function(_):
    learning.reset()


def test_neutral_with_no_history():
    assert learning.learned_success("coding", "m1") == 0.5


def test_record_biases_toward_successful_model():
    for _ in range(8):
        learning.record("coding", "good", success=True)
    for _ in range(8):
        learning.record("coding", "bad", success=False)
    assert learning.learned_success("coding", "good") > 0.7
    assert learning.learned_success("coding", "bad") < 0.3


def test_scoped_per_category():
    learning.record("coding", "m", success=True)
    # A different category has no history for the same model → neutral.
    assert learning.learned_success("writing", "m") == 0.5


def test_bounded_store_evicts_least_informative():
    # Fill past the cap; the single-sample entries are dropped first.
    learning._MAX_MODELS_PER_CAT  # exists
    big = "anchor"
    for _ in range(5):
        learning.record("coding", big, success=True)
    for i in range(learning._MAX_MODELS_PER_CAT + 10):
        learning.record("coding", f"m{i}", success=True)
    cat = learning._STATS["coding"]
    assert len(cat) <= learning._MAX_MODELS_PER_CAT
    assert big in cat        # the multi-sample anchor survived eviction


def test_persistence_round_trip():
    learning.record("coding", "m1", success=True)
    learning.record("coding", "m1", success=True)
    learning.record("coding", "m1", success=True)
    before = learning.learned_success("coding", "m1")
    assert before > 0.5
    prefs: dict = {}
    learning.export_to(prefs)
    assert "llm_routing_stats" in prefs
    learning.reset()
    assert learning.learned_success("coding", "m1") == 0.5   # cleared
    learning.load_from(prefs)
    # Restored history → matches the pre-reset value.
    assert learning.learned_success("coding", "m1") == before


def test_record_never_raises_on_none_key():
    learning.record("coding", None, success=True)   # no-op, no crash
    assert learning.learned_success("coding", None) == 0.5
