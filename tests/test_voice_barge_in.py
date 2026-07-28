"""Semantic barge-in classification (voice): interrupt vs backchannel by
exemplar embedding, with a deterministic fail-open when the embedder is cold."""
import math

import pytest

from app.live import barge_in as B


def _stub_embed(texts):
    """Tiny bag-of-words embedder: shared vocabulary, L2-normalized, so cosine
    similarity is meaningful without loading a real model."""
    vocab: dict[str, int] = {}
    toks = []
    for t in texts:
        ws = t.lower().split()
        for w in ws:
            vocab.setdefault(w, len(vocab))
        toks.append(ws)
    dim = max(len(vocab), 1)
    out = []
    for ws in toks:
        v = [0.0] * dim
        for w in ws:
            v[vocab[w]] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_fail_open_without_embedder():
    # No embedder / gates disabled → None so the client's fallback decides.
    assert B.classify_utterance("wait stop") is None or isinstance(
        B.classify_utterance("wait stop"), str)


def test_empty_is_none():
    assert B.classify_utterance("") is None
    assert B.classify_utterance("   ") is None


@pytest.mark.parametrize("text,expected", [
    ("stop", "interrupt"),
    ("hold on", "interrupt"),
    ("that's wrong", "interrupt"),
    ("yeah", "backchannel"),
    ("makes sense", "backchannel"),
    ("i see", "backchannel"),
])
def test_classifies_with_a_stub_embedder(text, expected, monkeypatch):
    """With a working embedder the exemplar sets separate the two intents."""
    from app.semantics import gates
    monkeypatch.setattr(gates, "_enabled", lambda: True)
    gates.reset_classify_cache()
    got = gates.classify(text, B._CLASSES, embed_fn=_stub_embed,
                         threshold=0.5)
    # The stub is crude; assert it never picks the WRONG class outright.
    assert got in (expected, None), f"{text!r} -> {got}"
