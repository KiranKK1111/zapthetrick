"""Generic semantic gates (2026-07-09) — orchestration decisions by exemplar
EMBEDDINGS, not hardcoded cue lists.

Every yes/no orchestration question ("is this a produce-a-document ask?",
"is this a question about the candidate themselves?", "is this an implicit
probe?") is expressed as DATA — positive and negative exemplar phrasings —
and answered by cosine similarity against the already-loaded bge-m3 embedder
(`app.rag.embedder`). Extending coverage means adding a sentence, not a regex.

Design (mirrors `app/clarify/intent_semantic.py`):
  • FAIL-OPEN: embedder unavailable / still warming → `score()` returns None
    and callers keep their deterministic fast-path verdict. The old cue lists
    survive ONLY as zero-latency fast-paths — the semantic layer is the
    authority for everything they miss.
  • Cold-start protection: the model is never loaded synchronously inside a
    request; while warming, gates return None.
  • Injectable `embed_fn` for tests; exemplar matrices cached per gate for
    the real embedder.
  • Thresholds are config-overridable (`semantic_gates.thresholds` mapping).
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

log = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

# ---------------------------------------------------------------------------
# Gate definitions — DATA, not rules. Edit/extend freely.
# ---------------------------------------------------------------------------
GATES: dict[str, dict] = {
    # "Produce a downloadable document for me" vs merely mentioning documents.
    "document_request": {
        "threshold": 0.62,
        "positives": [
            "generate a document on kafka basics",
            "create a document about our api design",
            "i want this as a downloadable file",
            "export this conversation as a file",
            "put this answer in a document",
            "make me a pdf of this",
            "give me a word document with the summary",
            "i need a soft copy i can download",
            "save this as a file for me",
            "turn this into a downloadable report",
            "prepare a document i can share with my team",
            "give me this in a file i can keep",
            "write this up as a formal document",
            "put together a document on the process for me",
            "draft a document covering the setup steps",
            "can you document it",
            "document this for me",
            "document the above for me",
            "can you document this conversation",
            "make a document out of this answer",
            "document it so i can share it",
        ],
        "negatives": [
            "summarize the document i uploaded",
            "what does this document say",
            "what is in the document i uploaded",
            "how do i create a document in python-docx",
            "how do i generate a pdf in code",
            "explain the document structure of mongodb",
            "review my document below",
            "parse this file and tell me what's in it",
            "what is markdown",
            "give me a quick summary here in chat",
            "explain this to me",
            "give me a report on the incident",
            "give me a report of what happened",
            "give me this in a tabular format",
            "can you get me in a tabular format",
            "show it as a table",
            "put the answer in a table",
            "give me this as bullet points",
            "list it out in columns",
            "format the response as a table",
            "write a login api",
            "write a function to parse the data",
            "create a file to store data in your code",
            "i need a presentation layer for my app",
            "the presentation layer of the architecture",
            "show me the code",
            "document this function with docstrings",
            "add documentation comments to my code",
            "document the api endpoints in the source code",
        ],
    },
    # "Package the project into an archive" vs coding questions about zips.
    "archive_request": {
        "threshold": 0.64,
        "positives": [
            "zip the project for me",
            "package the code for download",
            "bundle everything up so i can download it",
            "compress the whole codebase into one file",
            "give me an archive of the project",
            "download the entire source as one file",
            "i want the whole project as a single download",
            "wrap it all up in a zip",
        ],
        "negatives": [
            "how do i zip files in java",
            "how do i compress a string in python",
            "unzip this archive and explain what's inside",
            "what is a tarball",
            "why is my zip file corrupted",
            "explain how gzip compression works",
            "show me the code you wrote",
            "give me a report",
            "what is in the document i uploaded",
        ],
    },
    # Triage verification: does the message name a deliverable artifact?
    "artifact_delivery": {
        "threshold": 0.60,
        "positives": [
            "as a pdf please",
            "in excel format",
            "give me a downloadable file",
            "export it for me",
            "i want to download this",
            "attach it as a document",
            "make it a spreadsheet i can open",
            "send it to me as a file",
        ],
        "negatives": [
            "give me a report on the incident",
            "summarize this for me",
            "give me the details",
            "write it out here in chat",
            "just tell me the answer",
        ],
    },
    # LIVE: is the interviewer asking about the CANDIDATE themselves?
    "profile_question": {
        "threshold": 0.60,
        "positives": [
            "tell me about yourself",
            "walk me through your background",
            "what have you worked on recently",
            "describe your current role and responsibilities",
            "why should we hire you",
            "what are your strengths and weaknesses",
            "take me through your resume",
            "which projects are you most proud of",
            "what's your experience with kafka",
            "how many years of experience do you have",
            "tell us about a project you led",
            "what did you do at your last company",
            "why are you leaving your current job",
        ],
        "negatives": [
            "what is kafka",
            "explain dependency injection",
            "how does a hash map work",
            "design a url shortener",
            "what's the difference between tcp and udp",
            "how would you scale a database",
            "write a function to reverse a string",
        ],
    },
    # LIVE promotion: imperative/probing utterances that expect an answer
    # even without a question mark.
    "implicit_request": {
        "threshold": 0.62,
        "positives": [
            "walk me through your approach",
            "talk to me about scaling",
            "i'm curious about your reasoning here",
            "let's discuss the tradeoffs",
            "take me through the design",
            "share your thoughts on this",
            "elaborate on that a bit",
            "give me an example of that",
            "i'd love to hear how you'd handle it",
        ],
        "negatives": [
            "that makes sense",
            "okay let's move on",
            "thanks, that's all i needed",
            "i see, interesting",
            "let me check my notes",
            "we're running a bit short on time",
            "great, sounds good",
        ],
    },
    # LIVE promotion: hypothetical scenario probes.
    "hypothetical_scenario": {
        "threshold": 0.62,
        "positives": [
            "suppose one of your services goes down",
            "let's say we have a million users",
            "imagine your api is suddenly slow",
            "what if the database crashes mid-transaction",
            "consider a scenario where traffic doubles overnight",
            "picture this: the cache is completely cold",
            "say your queue starts backing up",
        ],
        "negatives": [
            "i suppose that's fine",
            "we assume standard latency here",
            "i imagine that took a while to build",
            "let's say goodbye to that idea",
            "suppose so",
        ],
    },
    # Conversation topic-shift: the user is starting a NEW subject (reset the
    # topic-scoped state) vs continuing the current thread.
    "topic_shift": {
        "threshold": 0.6,
        "positives": [
            "let's talk about something else",
            "new topic",
            "changing the subject",
            "on a completely different note",
            "forget that, let's discuss something new",
            "moving on to a different question",
            "unrelated question for you",
            "let's switch gears",
        ],
        "negatives": [
            "can you explain that a bit more",
            "what about the second option",
            "and how does that part work",
            "tell me more about it",
            "continue from where you left off",
            "go on",
            "what do you mean by that",
            "can you give an example of that",
        ],
    },
    # Read-only lookup: EXPLAIN / review existing content vs WRITE / build new.
    "read_only": {
        "threshold": 0.6,
        "positives": [
            "explain this code to me",
            "review the code below",
            "what does this function do",
            "walk me through this snippet",
            "analyze what this does",
            "tell me what's wrong with this code",
            "help me understand this",
            "what is this error saying",
        ],
        "negatives": [
            "write a function to parse the data",
            "build me a rest api",
            "create a python script for this",
            "implement a login flow",
            "generate the code for a todo app",
            "add a caching layer to my service",
            "make a new component for the form",
        ],
    },
    # Genuine "write me code" request (→ ask for a language when none is named)
    # vs a statement, a rendering/follow-up ("in a tabular format"), an opinion,
    # or a knowledge question that a classifier may mislabel as code generation.
    # This gate decides whether the deterministic "which language?" card is even
    # appropriate; a turn that already names a language never reaches it.
    "code_request": {
        "threshold": 0.60,
        "positives": [
            "write a program to reverse a string",
            "write a function to sort a list",
            "create a python script that reads a csv",
            "build a rest api for user management",
            "implement binary search",
            "give me the code for a login form",
            "code a snake game",
            "generate a regex that matches emails",
            "make a cli tool to rename files",
            "write me a sql query to join these tables",
            "can you write a function for this",
            "implement quicksort for me",
            "build a web scraper for this site",
            "write a class to represent a bank account",
            "create an endpoint that returns json",
            "can you code this up",
            "write the algorithm for this",
            "give me a script to automate this",
        ],
        "negatives": [
            "i don't want pin and section",
            "can you get me in a tabular format",
            "show it as a table",
            "give me this as bullet points",
            "put it in columns",
            "what is the difference between monolith and microservices",
            "explain how kafka works",
            "summarize the above",
            "give me more details on this",
            "what does this code do",
            "review my code below",
            "why does this happen",
            "make it shorter",
            "rewrite the answer in simpler terms",
            "what should i learn next",
            "is python better than java",
            "continue from where you left off",
            "put this in a document",
            "what is a hash map",
        ],
    },
}

# Cached (positives_matrix, negatives_matrix) per gate for the REAL embedder.
_CACHE: dict[str, tuple[object, object]] = {}


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    from app.rag.embedder import embed
    return embed(list(texts))


def _ready(embed_fn: EmbedFn | None) -> bool:
    if embed_fn is not None:
        return True
    try:
        from app.rag import embedder as _emb
        if not _emb.is_ready():
            _emb.ensure_loading_in_background()
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _matrices(gate: str, embed_fn: EmbedFn | None):
    import numpy as np
    spec = GATES[gate]
    if embed_fn is not None:
        pos = np.asarray(embed_fn(spec["positives"]), dtype="float32")
        neg = (np.asarray(embed_fn(spec["negatives"]), dtype="float32")
               if spec.get("negatives") else None)
        return pos, neg
    if gate not in _CACHE:
        pos = np.asarray(_default_embed(spec["positives"]), dtype="float32")
        neg = (np.asarray(_default_embed(spec["negatives"]), dtype="float32")
               if spec.get("negatives") else None)
        _CACHE[gate] = (pos, neg)
    return _CACHE[gate]


def reset_cache() -> None:
    _CACHE.clear()


def _enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.semantic_gates, "enabled", True))
    except Exception:  # noqa: BLE001
        return True


def threshold_for(gate: str) -> float:
    base = float(GATES[gate].get("threshold", 0.62))
    try:
        from app.core.config_loader import cfg
        overrides = getattr(cfg.semantic_gates, "thresholds", {}) or {}
        return float(overrides.get(gate, base))
    except Exception:  # noqa: BLE001
        return base


def margin_for(gate: str) -> float:
    """Required positive-minus-negative cosine margin. bge-m3 similarities
    sit high for ANYTHING vaguely on-topic, so discrimination comes from the
    margin against adversarial negatives, not the absolute threshold."""
    base = float(GATES[gate].get("margin", 0.04))
    try:
        from app.core.config_loader import cfg
        overrides = getattr(cfg.semantic_gates, "margins", {}) or {}
        return float(overrides.get(gate, base))
    except Exception:  # noqa: BLE001
        return base


def evaluate(gate: str, text: str,
             *, embed_fn: EmbedFn | None = None
             ) -> tuple[bool, float] | None:
    """(matches, best_positive_similarity) — or None when the embedder is
    unavailable (caller keeps its deterministic verdict). Matching requires
    BOTH the absolute threshold AND a margin over the nearest negative."""
    t = (text or "").strip().lower()
    if not t or gate not in GATES or not _enabled():
        return None
    if not _ready(embed_fn):
        return None
    try:
        import numpy as np
        pos, neg = _matrices(gate, embed_fn)
        ef = embed_fn or _default_embed
        v = np.asarray(ef([t])[0], dtype="float32")
        best_pos = float(np.max(pos @ v))
        ok = best_pos >= threshold_for(gate)
        if ok and neg is not None:
            best_neg = float(np.max(neg @ v))
            ok = (best_pos - best_neg) >= margin_for(gate)
        return ok, best_pos
    except Exception as exc:  # noqa: BLE001 — fail-open, deterministic wins
        log.info("semantic gate '%s' unavailable (%s)", gate, exc)
        return None


def score(gate: str, text: str,
          *, embed_fn: EmbedFn | None = None) -> float | None:
    """Best-positive similarity when the gate MATCHES; a value pinned below
    the threshold when it doesn't; None when the embedder is unavailable."""
    res = evaluate(gate, text, embed_fn=embed_fn)
    if res is None:
        return None
    ok, sim = res
    return sim if ok else min(sim, threshold_for(gate) - 0.01)


def matches(gate: str, text: str,
            *, embed_fn: EmbedFn | None = None) -> bool | None:
    """True/False when the embedder answered; None when unavailable."""
    res = evaluate(gate, text, embed_fn=embed_fn)
    return None if res is None else res[0]


# Cached exemplar matrices for multi-class classifiers (keyed by cache_key).
_CLASSIFY_CACHE: dict[str, dict] = {}


def classify(query: str, classes: dict, *,
             embed_fn: EmbedFn | None = None,
             threshold: float = 0.5,
             cache_key: str | None = None) -> str | None:
    """Semantic nearest-class classification — the multi-class counterpart of
    `matches`. Embeds ``query`` and each class's exemplars and returns the class
    whose nearest exemplar is most similar (>= ``threshold``), else None.

    Fail-open: None when semantic gates are disabled, the embedder is unavailable,
    or nothing clears the bar — so callers keep a deterministic FALLBACK. This is
    how intent is understood SEMANTICALLY (generalizing to paraphrases) rather
    than by hardcoded rules; the rules remain only as the cold-start safety net.
    """
    t = (query or "").strip().lower()
    if not t or not classes or not _enabled() or not _ready(embed_fn):
        return None
    try:
        import numpy as np
        ef = embed_fn or _default_embed
        if cache_key and embed_fn is None and cache_key in _CLASSIFY_CACHE:
            mats = _CLASSIFY_CACHE[cache_key]
        else:
            mats = {cls: np.asarray(ef(list(ex)), dtype="float32")
                    for cls, ex in classes.items() if ex}
            if cache_key and embed_fn is None:
                _CLASSIFY_CACHE[cache_key] = mats
        v = np.asarray(ef([t])[0], dtype="float32")
        best_cls, best_sim = None, -1.0
        for cls, mat in mats.items():
            sim = float(np.max(mat @ v))
            if sim > best_sim:
                best_sim, best_cls = sim, cls
        return best_cls if best_sim >= threshold else None
    except Exception as exc:  # noqa: BLE001 — fail-open, deterministic wins
        log.info("semantic classify unavailable (%s)", exc)
        return None


def reset_classify_cache() -> None:
    _CLASSIFY_CACHE.clear()


__all__ = ["GATES", "score", "matches", "classify", "threshold_for",
           "reset_cache", "reset_classify_cache"]
