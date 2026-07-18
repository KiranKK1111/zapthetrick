"""Phase 3 — Shared Cognitive Core, CHAT-side wiring.

Covers the previously-🟡 chat items brought to green:
  #7  Evidence graph + per-source trust   (app/rag/provenance.py)
  #11 Cross-source contradiction           (app/rag/conflict.py)
  #12 Human-interaction model + density    (app/understanding/interaction.py)
  #18 Hybrid search + MMR (wiring check)    (app/rag/retriever.py + mmr.py)
  #21 Document vision (layout/chart)        (app/rag/doc_vision.py)

Plus route-level wiring assertions (routes_agents consumes each module).
Deterministic + offline — no LLM, no DB.
"""
from __future__ import annotations

import inspect

import pytest


# ── #12 interaction / density selection engine ───────────────────────────────
from app.understanding import interaction as IX


class TestInteractionEngine:
    def test_comparison_selects_table(self):
        p = IX.select("compare kafka vs rabbitmq")
        assert p.shape == IX.COMPARISON
        assert p.action == IX.PROCEED
        assert "table" in p.shape_directive().lower()

    def test_how_to_selects_steps(self):
        p = IX.select("how do I set up postgres replication step by step")
        assert p.shape == IX.STEPS
        assert "number" in p.shape_directive().lower()

    def test_diagram_cue(self):
        p = IX.select("draw the architecture of the system as a diagram")
        assert p.shape == IX.DIAGRAM
        assert "mermaid" in p.shape_directive().lower()

    def test_missing_required_asks(self):
        p = IX.select("write a program", missing_required=["language"])
        assert p.action == IX.ASK

    def test_summarize_move(self):
        p = IX.select("give me a tl;dr of this thread")
        assert p.action == IX.SUMMARIZE

    def test_plain_prose_no_directive(self):
        p = IX.select("what is a deadlock?")
        assert p.shape == IX.PROSE and p.action == IX.PROCEED
        assert p.shape_directive() == ""

    def test_summarize_downgrades_heavy_shape(self):
        # A summary of a how-to shouldn't force a numbered-steps layout.
        p = IX.select("summarize how to install docker")
        assert p.action == IX.SUMMARIZE and p.shape not in (IX.STEPS, IX.DIAGRAM)

    def test_never_raises(self):
        assert IX.select(None).action == IX.PROCEED  # type: ignore[arg-type]


# ── #7 evidence graph + per-source trust ─────────────────────────────────────
from app.rag import provenance as PROV


class _Mem:
    def __init__(self, content, importance=0.7):
        self.content = content
        self.importance = importance


class TestProvenance:
    def test_per_source_trust_ordering(self):
        srcs = PROV.assemble(
            memory=[_Mem("decision: use postgres", 0.9)],
            episodes=[{"question": "how do I deploy?"}],
            kg_names=["Postgres"], kg_relations=["Postgres backs the API"],
            retrieval=[{"content": "Postgres is the store", "score": 0.8}])
        kinds = {s.kind for s in srcs}
        assert kinds == {"memory", "episode", "kg", "document"}
        # Document/retrieval evidence outranks a weak associative episode.
        doc = next(s for s in srcs if s.kind == "document")
        ep = next(s for s in srcs if s.kind == "episode")
        assert doc.trust > ep.trust

    def test_grounding_block_shape(self):
        srcs = PROV.assemble(memory=[_Mem("uses redis")])
        block = PROV.grounding_block(srcs)
        assert block["count"] == 1
        assert 0.0 <= block["trust"] <= 1.0
        assert block["sources"][0]["kind"] == "memory"

    def test_empty_is_none(self):
        assert PROV.grounding_block(PROV.assemble()) is None

    def test_kg_relation_support_raises_trust(self):
        backed = PROV.from_kg(["Kafka"], ["Kafka streams events"])
        bare = PROV.from_kg(["Kafka"], [])
        assert backed[0].trust >= bare[0].trust


# ── #11 cross-source contradiction ───────────────────────────────────────────
from app.rag import conflict as CONFLICT


class TestConflict:
    def test_value_conflict(self):
        cs = CONFLICT.detect(["database: postgres"], ["database: mysql"])
        assert cs and cs[0].kind == "value"

    def test_negation_flip(self):
        cs = CONFLICT.detect(
            ["we should use redis for caching"],
            ["we should not use redis for caching"])
        assert cs and cs[0].kind == "negation"

    def test_no_false_positive_on_agreement(self):
        assert CONFLICT.detect(["database: postgres"], ["database: postgres"]) == []

    def test_unrelated_no_conflict(self):
        assert CONFLICT.detect(["use postgres"], ["deploy on friday"]) == []

    def test_answer_scan(self):
        # Cross-source: committed memory vs a sentence in the produced answer.
        answer = "We will not use redis for caching. It adds ops overhead."
        cs = CONFLICT.detect(["we will use redis for caching"],
                             CONFLICT.split_sentences(answer))
        assert any(c.kind == "negation" for c in cs)

    def test_never_raises(self):
        assert CONFLICT.detect(None, None) == []


# ── #21 document vision — layout + chart understanding ───────────────────────
from app.rag import doc_vision as DV


_DOC = """# Quarterly Report

## Revenue

| Quarter | Revenue | Growth |
|---------|---------|--------|
| Q1      | 100     | 5%     |
| Q2      | 110     | 10%    |
| Q3      | 115     | 4%     |
| Q4      | 130     | 13%    |

Some prose here.

![chart](img.png)

```python
print("hi")
```
"""


class TestDocVision:
    def test_layout_skeleton(self):
        lay = DV.analyze_layout(_DOC)
        assert lay.max_depth == 2
        assert lay.tables == 1
        assert lay.figures == 1
        assert lay.code_blocks == 1
        assert any(h["text"] == "Revenue" for h in lay.headings)

    def test_chart_understanding(self):
        lay = DV.analyze_layout(_DOC)
        assert lay.charts, "a numeric table should be understood as a chart"
        ch = lay.charts[0]
        assert "Revenue" in ch.numeric_columns
        # Revenue rises 100 -> 130 monotonically → upward trend, peak at Q4.
        assert "upward" in ch.summary
        assert "130" in ch.summary and "Q4" in ch.summary

    def test_code_block_content_not_parsed_as_table(self):
        md = "```\n| not | a | table |\n| --- | --- | --- |\n| a | b | c |\n```"
        assert DV.analyze_layout(md).tables == 0

    def test_no_charts_for_text_table(self):
        md = "| Name | Role |\n| --- | --- |\n| Ann | Dev |\n| Bo | PM |"
        lay = DV.analyze_layout(md)
        assert lay.tables == 1 and lay.charts == []

    def test_never_raises(self):
        assert DV.analyze_layout(None).as_dict()["tables"] == 0  # type: ignore[arg-type]


# ── #18 hybrid + MMR wiring (import/reference proof) ─────────────────────────
class TestMmrWired:
    def test_retriever_reads_use_mmr_and_imports_mmr_filter(self):
        from app.rag import retriever
        src = inspect.getsource(retriever)
        assert "mmr_filter" in src
        assert "use_mmr" in src

    def test_mmr_filter_drops_near_duplicate(self):
        from app.rag.mmr import mmr_filter
        hits = [
            {"content": "kafka is a distributed event streaming platform", "score": 0.9},
            {"content": "kafka is a distributed event streaming platform.", "score": 0.88},
            {"content": "rabbitmq is a traditional message broker", "score": 0.6},
        ]
        picked = mmr_filter("streaming brokers", hits, top_k=3, lambda_=0.5)
        texts = [h["content"] for h in picked]
        assert any("rabbitmq" in t for t in texts)
        # The two near-identical kafka lines should not BOTH survive.
        assert sum(1 for t in texts if t.startswith("kafka")) == 1


# ── route-level wiring: routes_agents consumes each module ────────────────────
class TestRouteWiring:
    def test_routes_agents_consumes_phase3_modules(self):
        from app.api import routes_agents
        src = inspect.getsource(routes_agents)
        # TurnState build + consume (directive shaping).
        assert "TurnState.from_assessment" in src
        assert "answer_directive" in src
        assert 'extras_base["turn_state"]' in src
        # Interaction selection engine.
        assert "interaction as _ix" in src or "from app.understanding import interaction" in src
        # Evidence provenance + conflict + confidence band on the envelope.
        assert "provenance as _prov" in src
        assert "conflict as _cflt" in src
        assert "confidence_band=_conf_band" in src
        assert "grounding=_grounding" in src
