"""Tests for delivery tracking + true said-state (vNext §4.14, Stage 7 Component I)."""
from __future__ import annotations

import app.live.delivery as D


# ---- alignment: the delivery cursor ---------------------------------------
def test_full_delivery_completes():
    s = D.align_delivery(
        "Kafka is a distributed log. It uses partitions for parallelism.",
        "kafka is a distributed log it uses partitions for parallelism")
    assert s.delivered_ratio == 1.0
    assert s.completed is True
    assert s.improvised == []
    assert s.cursor == s.displayed_words


def test_interruption_only_delivered_prefix_counts():
    displayed = "Kafka is a distributed log. It uses partitions for parallelism."
    s = D.align_delivery(displayed, "kafka is a distributed log")
    assert 0.4 < s.delivered_ratio < 0.6      # ~half the script spoken
    assert s.completed is False
    # The unspoken tail is NOT part of the delivered text.
    assert "partitions" not in s.delivered_text.lower()
    assert s.delivered_text.strip() == "Kafka is a distributed log"


def test_nothing_spoken_yet():
    s = D.align_delivery("Kafka is a log.", "")
    assert s.delivered_ratio == 0.0
    assert s.cursor == 0
    assert s.delivered_text == ""
    assert s.completed is False


def test_empty_displayed_is_safe():
    s = D.align_delivery("", "anything at all here")
    assert s.displayed_words == 0
    assert s.delivered_ratio == 0.0


# ---- improvisation: off-script speech enters said-state -------------------
def test_improvised_trailing_run_captured():
    s = D.align_delivery(
        "Kafka is a distributed log.",
        "kafka is a distributed log and i deployed it at scale in production")
    assert s.improvised
    assert "deployed it at scale" in s.improvised[0]
    assert D.improvised_claims(s)


def test_short_off_script_run_is_not_a_claim():
    # A 2-word aside is below the claim threshold → not counted.
    s = D.align_delivery("Kafka is a distributed log.",
                         "so yeah kafka is a distributed log")
    assert D.improvised_claims(s) == []


def test_said_text_is_delivered_plus_improvised():
    s = D.align_delivery(
        "Kafka is a distributed log.",
        "kafka is a distributed log and i ran it in production myself")
    said = D.said_text(s)
    assert "Kafka is a distributed log" in said
    assert "ran it in production" in said


def test_said_text_excludes_unspoken_tail():
    displayed = "First point here. Second point never spoken aloud today."
    s = D.align_delivery(displayed, "first point here")
    said = D.said_text(s)
    assert "First point here" in said
    assert "never spoken" not in said.lower()


# ---- record_delivery: gating + ledger feed --------------------------------
def test_record_delivery_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: False)
    recorded = []
    import app.live.canonical as C
    monkeypatch.setattr(C, "record_claim", lambda sid, t: recorded.append((sid, t)))
    s = D.record_delivery("sess-x", "Kafka is a log.", "kafka is a log")
    assert recorded == []                     # disabled → never touches the ledger
    assert isinstance(s, D.DeliveryState)     # still returns the alignment


def test_record_delivery_feeds_ledger_when_enabled(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)
    recorded = []
    import app.live.canonical as C
    monkeypatch.setattr(C, "record_claim", lambda sid, t: recorded.append(t))
    D.record_delivery(
        "sess-y",
        "Kafka is a distributed log.",
        "kafka is a distributed log and i personally scaled it in production")
    joined = " ".join(recorded)
    assert "Kafka is a distributed log" in joined          # delivered script
    assert "scaled it in production" in joined              # improvised claim


def test_record_delivery_interrupted_does_not_record_unspoken_tail(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)
    recorded = []
    import app.live.canonical as C
    monkeypatch.setattr(C, "record_claim", lambda sid, t: recorded.append(t))
    # Only a small fraction delivered (< 0.5) → the script is NOT recorded as said.
    D.record_delivery(
        "sess-z",
        "One two three four five six seven eight nine ten.",
        "one two")
    assert recorded == []


def test_never_raises_on_garbage():
    s = D.align_delivery(None, None)          # type: ignore[arg-type]
    assert s.delivered_ratio == 0.0


def test_as_dict_shape():
    s = D.align_delivery("Kafka is a log.", "kafka is a log")
    d = s.as_dict()
    assert set(d) == {"displayed_words", "delivered_words", "delivered_ratio",
                      "cursor", "completed", "improvised"}
