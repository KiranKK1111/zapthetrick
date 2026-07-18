"""Retriever — calls the production RAG pipeline.

P0, must finish under ~500ms warm / ~1500ms cold. Delegates to
`app.rag.retriever.retrieve` (hybrid Postgres FTS + vector + RRF +
cross-encoder rerank), which reads chunks from the `resume_chunks`
table and vectors from the configured [VectorStore].

Requires `extras.resume_id` and `extras.db_session` on the blackboard
to actually run. If either is missing — e.g. a free-form chat with no
attached resume — this writes an empty Evidence object and returns
quickly, letting the Persona agent fall back to its system prompt.
"""
from __future__ import annotations

from ..blackboard.board import Blackboard
from ..blackboard.schema import (
    KEY_EVIDENCE,
    KEY_QUESTION,
    Evidence,
    EvidenceChunk,
)
from ..blackboard.scheduler import P0
from .base import Agent


class RetrieverAgent(Agent):
    name = "retriever"
    priority = P0
    expected_latency_ms = 400
    reads = frozenset({KEY_QUESTION})
    writes = frozenset({KEY_EVIDENCE})

    # Code-graph tools are already covered by the dedicated code-evidence
    # source, so the generic tool executor skips them (no double work / cost).
    _CODE_TOOLS = frozenset({"code_search", "code_callers", "code_callees",
                             "code_impact", "code_file_structure"})

    async def run(self, board: Blackboard) -> None:
        import asyncio

        extras = board.get("extras", {}) or {}
        resume_id = extras.get("resume_id")
        conversation_id = extras.get("conversation_id")
        db_session = extras.get("db_session")
        question = board.get(KEY_QUESTION, "")

        if not question.strip():
            board.write(KEY_EVIDENCE, Evidence(), agent=self.name)
            return

        # All four sources are independent and I/O-bound, so run them
        # CONCURRENTLY — the tool executor's LLM call overlaps the RAG queries
        # instead of adding its latency on top. (Only resume RAG touches the
        # shared db_session; the rest use their own sessions, so this is safe.)
        async def _resume() -> list[EvidenceChunk]:
            if not (resume_id and db_session is not None):
                return []
            try:
                from ..rag.retriever import retrieve as rag_retrieve
                hits = await rag_retrieve(question, resume_id=resume_id, session=db_session)
                return [EvidenceChunk(text=h.text, source=h.section or "resume",
                                      score=h.score, parent_id=h.chunk_id) for h in hits]
            except Exception:  # noqa: BLE001
                return []

        async def _chat_docs() -> list[EvidenceChunk]:
            if not conversation_id:
                return []
            try:
                from ..rag.documents import retrieve_chat_hits
                dhits = await retrieve_chat_hits(str(conversation_id), question)
                return [EvidenceChunk(text=h["content"], source=h["filename"],
                                      score=h["score"], parent_id=None) for h in dhits]
            except Exception:  # noqa: BLE001
                return []

        async def _code_graph() -> list[EvidenceChunk]:
            if not conversation_id:
                return []
            try:
                from ..codegraph.retrieval import retrieve_code_evidence
                hits = await retrieve_code_evidence(str(conversation_id), question)
                return [EvidenceChunk(text=h["text"], source=h["source"],
                                      score=h["score"], parent_id=None) for h in hits]
            except Exception:  # noqa: BLE001
                return []

        async def _knowledge_graph() -> list[EvidenceChunk]:
            # §3.1 grounding: relations of the turn's entities from the persisted
            # content KG (conversation- or project-scoped) join the evidence set,
            # so the answer connects concepts the graph already knows. Cheap —
            # one row read + local traversal, no LLM.
            if not conversation_id:
                return []
            try:
                from ..core.config_loader import cfg
                if not getattr(cfg.advanced_rag, "use_knowledge_graph", False):
                    return []
                from ..rag.documents import load_conversation_kg
                from ..rag.kg_extract import relations_in_text
                kg = await load_conversation_kg(str(conversation_id))
                if kg is None:
                    return []
                triples = relations_in_text(kg, question, limit=6)
                if not triples:
                    return []
                return [EvidenceChunk(
                    text="Known relations (knowledge graph): "
                         + "; ".join(triples),
                    source="knowledge_graph", score=0.6, parent_id=None)]
            except Exception:  # noqa: BLE001
                return []

        async def _tools() -> list[EvidenceChunk]:
            try:
                from ..core.config_loader import cfg
                if not cfg.advanced_rag.use_tool_executor:
                    return []
                import json as _json

                from ..tools.executor import run_relevant_tools
                ctx = {"conversation_id": str(conversation_id)} if conversation_id else {}
                # §4 Intent Profile Registry: when enabled, the intent's profile
                # constrains which tools may run (allow-list + excludes + cap).
                # The retriever runs concurrently with the Planner, so it self-
                # classifies with the FAST regex intent (no ordering dependency).
                _allow = None
                _exclude = set(self._CODE_TOOLS)
                _max_tools = 2
                try:
                    from ..clarify import intent_profiles as _ip
                    if _ip.enabled():
                        from ..clarify.intent_pipeline import detect_intent
                        _prof = _ip.resolve(detect_intent(question))
                        if _prof.tools is not None:
                            if not _prof.tools:
                                return []  # profile allows no tools for this intent
                            _allow = set(_prof.tools)
                        _exclude |= set(_prof.exclude_tools)
                        _max_tools = _prof.max_tools
                except Exception:  # noqa: BLE001 — fail-open to today's behavior
                    _allow, _exclude, _max_tools = None, set(self._CODE_TOOLS), 2
                out: list[EvidenceChunk] = []
                for r in await run_relevant_tools(
                        question, context=ctx, allow=_allow,
                        exclude=_exclude, max_tools=_max_tools):
                    body = r["result"]
                    text = body if isinstance(body, str) else _json.dumps(body)[:4000]
                    out.append(EvidenceChunk(text=f"[{r['tool']}] {text}",
                                             source=f"tool:{r['tool']}", score=0.7,
                                             parent_id=None))
                return out
            except Exception:  # noqa: BLE001
                return []

        groups = await asyncio.gather(_resume(), _chat_docs(), _code_graph(),
                                      _knowledge_graph(), _tools())
        chunks = [c for grp in groups for c in grp]

        board.write(
            KEY_EVIDENCE,
            Evidence(
                chunks=chunks,
                sources=[c.source for c in chunks],
                confidences=[c.score for c in chunks],
            ),
            agent=self.name,
        )
