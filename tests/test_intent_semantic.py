"""Semantic intent classifier plumbing (app/clarify/intent_semantic).

The real bge-m3 model is too heavy to load in CI, so these tests inject a
deterministic fake embedder to pin the MECHANICS (nearest-exemplar lookup,
label mapping, cosine via dot, empty handling, fail-open). Semantic QUALITY is
validated separately against the real model in the eval harness.
"""
from __future__ import annotations

import hashlib
import os

import pytest

np = pytest.importorskip("numpy")

from app.clarify import intent_semantic as S
from app.clarify.intent_pipeline import (
    INTENT_ARCHIVE, INTENT_CODE_GEN, INTENT_KNOWLEDGE,
)


def _fake_embed(texts):
    """Deterministic unit vector per string (hash-seeded). An identical string
    yields an identical vector → cosine 1.0, so an exact-exemplar query is its
    own nearest neighbour."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h * 4, dtype=np.uint8)[:64].astype("float32") - 127.5
        v /= (np.linalg.norm(v) or 1.0)
        out.append(v.tolist())
    return out


def test_exact_exemplar_returns_its_intent():
    for phrase, intent in (
        ("explain this code", INTENT_KNOWLEDGE),
        ("write a function to reverse a string", INTENT_CODE_GEN),
        ("zip the whole project", INTENT_ARCHIVE),
    ):
        label, score = S.classify(phrase, embed_fn=_fake_embed)
        assert label == intent, phrase
        assert score > 0.99, (phrase, score)


def test_empty_input_returns_none():
    assert S.classify("", embed_fn=_fake_embed) is None
    assert S.classify("   ", embed_fn=_fake_embed) is None


def test_fail_open_on_embedder_error():
    def boom(_texts):
        raise RuntimeError("model unavailable")
    assert S.classify("anything at all", embed_fn=boom) is None


def test_score_in_cosine_range():
    label, score = S.classify("implement binary search", embed_fn=_fake_embed)
    assert label in S.EXEMPLARS
    assert -1.0001 <= score <= 1.0001


def test_every_intent_has_exemplars():
    for intent, phrases in S.EXEMPLARS.items():
        assert phrases, intent
        assert all(isinstance(p, str) and p.strip() for p in phrases), intent


# --- tiered reconciliation (detect_intent_smart) ---------------------------

def _enable_semantic(monkeypatch, *, primary=0.50):
    """Force `cfg.semantic_intent.enabled` on without persisting to disk."""
    import app.core.config_loader as C
    base = C.get_config()
    patched = base.model_copy(update={
        "semantic_intent": base.semantic_intent.model_copy(update={
            "enabled": True, "primary_threshold": primary,
        })
    })
    monkeypatch.setattr(C, "_config", patched)


def test_smart_is_pure_regex_when_disabled(monkeypatch):
    # conftest disables semantic by default → identical to regex, no model load.
    from app.clarify.intent_pipeline import detect_intent, detect_intent_smart
    for q in ("write a function to reverse a string", "explain how kafka works",
              "build me a web app"):
        assert detect_intent_smart(q) == detect_intent(q)


def test_semantic_is_primary_above_threshold(monkeypatch):
    # A confident semantic verdict decides directly — the keyword regex does not
    # participate, even when it would say something different.
    from app.clarify import intent_semantic
    from app.clarify.intent_pipeline import (
        detect_intent, detect_intent_smart, INTENT_CODE_GEN, INTENT_DEBUGGING)
    _enable_semantic(monkeypatch)
    q = "write a function to reverse a string"          # regex → code_generation
    assert detect_intent(q) == INTENT_CODE_GEN
    monkeypatch.setattr(intent_semantic, "classify",
                        lambda text, embed_fn=None: (INTENT_DEBUGGING, 0.62))
    assert detect_intent_smart(q) == INTENT_DEBUGGING   # semantic is authoritative


def test_low_confidence_defers_to_regex_net(monkeypatch):
    from app.clarify import intent_semantic
    from app.clarify.intent_pipeline import (
        detect_intent, detect_intent_smart, INTENT_CODE_GEN,
        INTENT_DEBUGGING, INTENT_PROJECT_BUILD, INTENT_UNKNOWN)
    _enable_semantic(monkeypatch)
    # Below primary_threshold + regex HAS an opinion → the regex net wins.
    q = "write a function to reverse a string"          # regex → code_generation
    monkeypatch.setattr(intent_semantic, "classify",
                        lambda text, embed_fn=None: (INTENT_DEBUGGING, 0.30))
    assert detect_intent_smart(q) == INTENT_CODE_GEN
    # Below threshold + regex has NO opinion → semantic best-guess is used.
    q2 = "xyzzy foobar plugh"                            # regex → unknown
    assert detect_intent(q2) == INTENT_UNKNOWN
    monkeypatch.setattr(intent_semantic, "classify",
                        lambda text, embed_fn=None: (INTENT_PROJECT_BUILD, 0.30))
    assert detect_intent_smart(q2) == INTENT_PROJECT_BUILD


def test_smart_fails_open_when_embedder_returns_none(monkeypatch):
    from app.clarify import intent_semantic
    from app.clarify.intent_pipeline import detect_intent, detect_intent_smart
    _enable_semantic(monkeypatch)
    monkeypatch.setattr(intent_semantic, "classify",
                        lambda text, embed_fn=None: None)   # embedder unavailable
    for q in ("write a function to reverse a string", "explain how kafka works"):
        assert detect_intent_smart(q) == detect_intent(q)


# --- real-model quality benchmark -------------------------------------------
# Runs automatically when the embedding model is ALREADY cached locally (dev
# machines that have bge-m3), and skips otherwise (clean/CI machines) so it never
# triggers a ~2GB download. Force-on with RUN_SEMANTIC_EVAL=1 regardless.
def _embed_model_cached() -> bool:
    if os.environ.get("RUN_SEMANTIC_EVAL") == "1":
        return True
    try:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            return False
        from app.core.config_loader import cfg
        model_id = (cfg.embeddings.model or "").strip()
        if not model_id:
            return False
        from huggingface_hub.constants import HF_HUB_CACHE
        folder = "models--" + model_id.replace("/", "--")
        return os.path.isdir(os.path.join(HF_HUB_CACHE, folder))
    except Exception:  # noqa: BLE001 — unknown → skip (safe default)
        return False


@pytest.mark.skipif(
    not _embed_model_cached(),
    reason="embedding model not cached locally (set RUN_SEMANTIC_EVAL=1 to force)",
)
def test_real_model_beats_regex_on_paraphrases(monkeypatch):
    _enable_semantic(monkeypatch)
    from app.clarify.intent_pipeline import detect_intent_smart
    # Cold-start protection makes classify() fail open to regex until the
    # model is READY (it never loads synchronously in-request any more) —
    # warm it explicitly so this test exercises the real model.
    from app.rag import embedder
    embedder.embed(["warmup"])
    # Paraphrases the keyword regex gets wrong or misses.
    cases = [
        ("why does my function blow up on an empty list", "debugging"),
        ("stand up a new nextjs app for me", "project_build"),
        ("add coverage for this handler", "test_generation"),
        ("how is a mutex different from a semaphore", "comparison"),
        ("sketch the high-level structure for a payments service", "design"),
        # regex-correct cases must not regress
        ("write a function to reverse a string", "code_generation"),
        ("explain how kafka works", "knowledge"),
    ]
    correct = sum(detect_intent_smart(q) == exp for q, exp in cases)
    assert correct >= len(cases) - 1, f"only {correct}/{len(cases)} correct"
