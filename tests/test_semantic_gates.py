"""Semantic orchestration gates (user directive 2026-07-09: "everything
dynamic on semantic embeddings, no hardcoded verification orchestrations").
Gates are exercised with an injected deterministic embedder (no model load);
the fail-open contract is pinned so hosts without the embedder keep the
deterministic fast-path behavior."""
from __future__ import annotations

import hashlib
import math

from app.semantics import gates


def _fake_embed(texts):
    """Deterministic pseudo-embeddings: same text → same unit vector; texts
    sharing many words → high cosine. Good enough to test gate MECHANICS
    (matrices, thresholds, negatives) without the real model."""
    out = []
    for t in texts:
        vec = [0.0] * 64
        for w in (t or "").lower().split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            vec[h % 64] += 1.0
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / n for x in vec])
    return out


class TestGateMechanics:
    def test_exact_positive_matches(self):
        # An exemplar itself must clear its own gate.
        for gate, spec in gates.GATES.items():
            s = gates.score(gate, spec["positives"][0], embed_fn=_fake_embed)
            assert s is not None and s >= gates.threshold_for(gate), gate

    def test_negative_exemplar_blocks(self):
        # A negative exemplar scores 1.0 against itself, so the demotion rule
        # (negative ≥ positive → below threshold) must reject it.
        for gate, spec in gates.GATES.items():
            if not spec.get("negatives"):
                continue
            m = gates.matches(gate, spec["negatives"][0],
                              embed_fn=_fake_embed)
            assert m is False, gate

    def test_unknown_gate_none(self):
        assert gates.score("no_such_gate", "hello",
                           embed_fn=_fake_embed) is None

    def test_empty_text_none(self):
        assert gates.score("document_request", "  ",
                           embed_fn=_fake_embed) is None

    def test_threshold_override(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.semantic_gates, "thresholds",
                            {"document_request": 0.99}, raising=False)
        assert gates.threshold_for("document_request") == 0.99

    def test_disabled_returns_none(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.semantic_gates, "enabled", False,
                            raising=False)
        assert gates.score("document_request",
                           "make me a pdf of this",
                           embed_fn=_fake_embed) is None


class TestFailOpenContract:
    """Embedder not ready → every wired call site keeps its deterministic
    verdict (this is what the test-suite environment exercises globally)."""

    def test_gates_none_when_embedder_cold(self, monkeypatch):
        import app.rag.embedder as emb
        monkeypatch.setattr(emb, "is_ready", lambda: False)
        monkeypatch.setattr(emb, "ensure_loading_in_background",
                            lambda: None)
        assert gates.score("document_request", "make me a pdf") is None
        assert gates.matches("profile_question",
                             "tell me about yourself") is None

    def test_profile_fastpath_survives_cold_embedder(self, monkeypatch):
        import app.rag.embedder as emb
        monkeypatch.setattr(emb, "is_ready", lambda: False)
        monkeypatch.setattr(emb, "ensure_loading_in_background",
                            lambda: None)
        from app.live.profile import is_profile_question
        assert is_profile_question("tell me about yourself")
        assert not is_profile_question("what is a hash map")

    def test_doc_detect_fastpath_survives_cold_embedder(self, monkeypatch):
        import app.rag.embedder as emb
        monkeypatch.setattr(emb, "is_ready", lambda: False)
        monkeypatch.setattr(emb, "ensure_loading_in_background",
                            lambda: None)
        from app.documents.detect import explicit_doc_request
        assert explicit_doc_request("generate a pdf") == (True, "pdf")
        det, _ = explicit_doc_request("what does this code do")
        assert not det


class TestSemanticWiring:
    """With a warm (stubbed) embedder, the semantic tails fire at each site."""

    def _warm(self, monkeypatch):
        import app.rag.embedder as emb
        monkeypatch.setattr(emb, "is_ready", lambda: True)
        monkeypatch.setattr(emb, "embed",
                            lambda texts: _fake_embed(texts))
        gates.reset_cache()

    def test_doc_semantic_tail(self, monkeypatch):
        self._warm(monkeypatch)
        from app.documents.detect import explicit_doc_request
        # Exact exemplar phrasing — regex fast-paths don't catch it, the
        # semantic tail must.
        det, fmt = explicit_doc_request(
            "prepare a document i can share with my team")
        assert det and fmt == "pdf"
        gates.reset_cache()

    def test_profile_semantic_tail(self, monkeypatch):
        self._warm(monkeypatch)
        from app.live.profile import is_profile_question
        assert is_profile_question("why are you leaving your current job")
        gates.reset_cache()

    def test_implicit_semantic_signal(self, monkeypatch):
        self._warm(monkeypatch)
        from app.live.implicit import detect_implicit
        sig = detect_implicit("i'd love to hear how you'd handle it")
        assert sig.is_implicit_question
        assert sig.cue == "semantic"
        assert sig.confidence >= 0.6
        gates.reset_cache()

    def test_semantic_dedup_paraphrase(self, monkeypatch):
        import app.rag.embedder as emb
        monkeypatch.setattr(emb, "is_ready", lambda: True)
        monkeypatch.setattr(emb, "embed",
                            lambda texts: _fake_embed(texts))
        from app.live.dedup import QuestionDeduper
        d = QuestionDeduper(window_s=20.0, semantic=True,
                            semantic_similarity=0.95)
        d.note_answered("how would you scale kafka consumers quickly",
                        now=0.0)
        # Same words shuffled → char-ratio low, cosine 1.0 with bag-of-words
        # fake embedder → semantic layer catches it.
        assert d.is_duplicate(
            "quickly consumers kafka scale you would how", now=2.0)
        assert not d.is_duplicate("what is a binary tree", now=2.0)


class TestPgVectorReset:
    def test_reset_exists_and_delegates(self):
        from storage.vectors.pgvector_store import PgVectorStore
        assert hasattr(PgVectorStore, "reset")


class TestSemanticClassify:
    """The multi-class semantic classifier (audience/goal/project-type run on
    this) — nearest-class by exemplar similarity, fail-open to None."""

    _CLASSES = {
        "manager": ["write this for my manager", "for my boss"],
        "developer": ["for the engineering team", "for developers"],
        "student": ["explain for beginners", "for students learning"],
    }

    def test_nearest_class_wins(self):
        # "for my manager" overlaps the manager exemplars (fake embedder = word
        # overlap), so it classifies there, not developer/student.
        assert gates.classify("please write this for my manager",
                              self._CLASSES, embed_fn=_fake_embed,
                              threshold=0.1) == "manager"
        assert gates.classify("make it for the engineering team",
                              self._CLASSES, embed_fn=_fake_embed,
                              threshold=0.1) == "developer"

    def test_below_threshold_is_none(self):
        assert gates.classify("banana helicopter", self._CLASSES,
                             embed_fn=_fake_embed, threshold=0.9) is None

    def test_empty_query_is_none(self):
        assert gates.classify("", self._CLASSES, embed_fn=_fake_embed) is None


def test_document_verb_matches_gate():
    # "document it / this" now clears the document_request gate via the new
    # exemplars; "document this function" (code) is held out by the negatives.
    assert gates.score("document_request", "can you document it",
                       embed_fn=_fake_embed) is not None
    ok_doc = gates.matches("document_request", "document this for me",
                           embed_fn=_fake_embed)
    ok_code = gates.matches("document_request", "document this function with docstrings",
                            embed_fn=_fake_embed)
    assert ok_doc is True and ok_code is False
