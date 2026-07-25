"""Stage-7 §8.3 — span citations: ground answer claims to source chunks."""
from __future__ import annotations

import pytest

from app.rag import citations as C

_CHUNKS = [
    {"content": "Apache Kafka is a distributed commit log that decouples "
                "producers from consumers for durable streaming.",
     "doc": "kafka-guide", "chunk_id": "c1"},
    {"content": "Topics in Kafka are partitioned, which lets a cluster scale "
                "horizontally across brokers.",
     "doc": "kafka-guide", "chunk_id": "c2"},
]


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "span_citations", True, raising=False)
    monkeypatch.setattr(cfg.advanced_rag, "span_citation_min_score", 0.4,
                        raising=False)


class TestBuildCitations:
    def test_grounds_a_supported_claim(self, _on):
        ans = "Kafka decouples producers from consumers using a commit log."
        cites = C.build_citations(ans, _CHUNKS)
        assert len(cites) == 1
        c = cites[0]
        assert c.doc == "kafka-guide" and c.chunk == "c1"
        assert "decouples producers from consumers" in c.quote
        assert c.index == 1

    def test_claim_span_points_into_the_answer(self, _on):
        ans = "Kafka decouples producers from consumers reliably."
        c = C.build_citations(ans, _CHUNKS)[0]
        s, e = c.claim_span
        assert ans[s:e].startswith("Kafka decouples")

    def test_quote_span_points_into_the_chunk(self, _on):
        ans = "Kafka decouples producers from consumers."
        c = C.build_citations(ans, _CHUNKS)[0]
        s, e = c.quote_span
        assert _CHUNKS[0]["content"][s:e] == c.quote

    def test_unsupported_claim_is_not_cited(self, _on):
        assert C.build_citations(
            "The sky is green and made entirely of cheese today.", _CHUNKS) == []

    def test_sequential_indices(self, _on):
        ans = ("Kafka decouples producers from consumers for durable streaming. "
               "Topics in Kafka are partitioned across brokers.")
        cites = C.build_citations(ans, _CHUNKS)
        assert [c.index for c in cites] == list(range(1, len(cites) + 1))
        assert len(cites) == 2                       # both claims grounded

    def test_picks_the_best_chunk(self, _on):
        # A partitioning claim should cite c2, not c1.
        ans = "Topics in Kafka are partitioned across brokers to scale."
        c = C.build_citations(ans, _CHUNKS)[0]
        assert c.chunk == "c2"

    def test_accepts_text_and_source_keys(self, _on):
        chunks = [{"text": "Kafka decouples producers from consumers.",
                   "source": "d1", "id": "x"}]
        c = C.build_citations("Kafka decouples producers from consumers.",
                              chunks)[0]
        assert c.doc == "d1" and c.chunk == "x"

    def test_disabled_returns_empty(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.advanced_rag, "span_citations", False,
                            raising=False)
        assert C.build_citations("Kafka decouples producers.", _CHUNKS) == []

    def test_empty_inputs(self, _on):
        assert C.build_citations("", _CHUNKS) == []
        assert C.build_citations("Kafka decouples producers.", []) == []

    def test_never_raises(self, _on):
        C.build_citations("x", [None, {"content": None}, "notadict"])  # type: ignore[list-item]


class TestGroundingShape:
    def test_grounding_dict_shape(self, _on):
        cites = C.build_citations(
            "Kafka decouples producers from consumers.", _CHUNKS)
        g = C.grounding_citations(cites)
        assert "citations" in g and g["citations"][0]["doc"] == "kafka-guide"
        assert set(g["citations"][0]) >= {"index", "claim_span", "doc", "chunk",
                                          "quote", "quote_span", "score"}

    def test_empty_when_no_citations(self):
        assert C.grounding_citations([]) == {}
