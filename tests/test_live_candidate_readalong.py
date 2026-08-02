"""Solo mode: the candidate reads the answer aloud WHILE it streams.

The reported failure: in solo mode the app hears the candidate speaking the
answer back, treats it as a new question, and answers its own words — talking
over the person it is supposed to be helping.

The cause was timing. `remember_answer` ran only after an answer COMPLETED, but
a read-along starts while the answer is still arriving. At that moment the echo
store held nothing for it, so the utterance matched nothing and went through as
a prompt.

A second, quieter failure sat behind it: someone who reads only the opening of a
long answer is not very similar to the WHOLE answer, so even correct timing
could score below threshold. Registering PREFIXES as they grow fixes both.

These tests use a deterministic stub embedder — the real one is a heavy model and
its exact cosine values are not the contract. What is under test is *when*
registration happens and *what* is matchable, not bge-m3's numerics.
"""
from __future__ import annotations

import math

import pytest

from app.live import echo as E

ANSWER = (
    "A hash map stores key value pairs using a hash function to place each key "
    "into a bucket. Lookups average constant time because the hash points "
    "straight at the bucket. Collisions are handled by chaining or by open "
    "addressing, and the table is resized once the load factor climbs."
)


@pytest.fixture(autouse=True)
def clean():
    E.forget_session("s1")
    yield
    E.forget_session("s1")


@pytest.fixture()
def stub_embedder(monkeypatch):
    """A bag-of-words unit vector: similar text ⇒ high cosine, without loading a
    real model. Deterministic, so a threshold assertion means something."""
    def embed_one(text: str):
        counts: dict[str, float] = {}
        for w in (text or "").lower().split():
            w = w.strip(".,()")
            if w:
                counts[w] = counts.get(w, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        # Fixed 512-slot projection so every vector is the same length.
        vec = [0.0] * 512
        for w, v in counts.items():
            vec[hash(w) % 512] += v / norm
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]

    import app.rag.embedder as emb
    monkeypatch.setattr(emb, "embed_one", embed_one)
    return embed_one


# ── The reported bug ────────────────────────────────────────────────────────

def test_a_read_along_is_recognised_while_the_answer_is_still_streaming(
        stub_embedder):
    """The bug, directly. Nothing has COMPLETED yet — only a prefix has streamed
    — and the candidate is already reading it aloud."""
    streamed = ""
    for word in ANSWER.split():
        streamed += word + " "
        E.remember_streaming("s1", streamed)
        if len(streamed) > 200:
            break

    spoken = streamed.strip()
    is_echo, sim = E.is_candidate_echo("s1", spoken)
    assert is_echo, (
        f"a read-along mid-stream was not recognised (similarity {sim:.2f}) — "
        "it would be transcribed and answered as a new question")


def test_reading_only_the_OPENING_of_a_long_answer_still_matches(stub_embedder):
    """The quieter half of the bug: a partial read is not very similar to the
    whole answer, so a single final-text entry can score under threshold."""
    E.remember_streaming("s1", ANSWER[:160])
    E.remember_streaming("s1", ANSWER[:320])
    E.remember_answer("s1", ANSWER)

    opening = ANSWER[:150]
    is_echo, sim = E.is_candidate_echo("s1", opening)
    assert is_echo, f"partial read scored only {sim:.2f}"


# ── It must not swallow real questions ──────────────────────────────────────

def test_a_genuine_follow_up_is_NOT_treated_as_an_echo(stub_embedder):
    """The failure mode that would be worse than the bug: suppressing a real
    question because it shares vocabulary with the answer on screen."""
    E.remember_streaming("s1", ANSWER[:200])
    E.remember_answer("s1", ANSWER)

    for question in (
        "And how would you handle collisions at scale",
        "What about thread safety",
        "Why not use a tree instead",
    ):
        is_echo, sim = E.is_candidate_echo("s1", question)
        assert not is_echo, f"suppressed a real question ({sim:.2f}): {question!r}"


def test_an_empty_session_matches_nothing(stub_embedder):
    assert E.is_candidate_echo("s1", "anything at all")[0] is False


# ── Cost and bookkeeping ────────────────────────────────────────────────────

def test_registration_is_throttled_not_per_token(stub_embedder):
    """Embedding on every token would be absurd. Growth-gated, so a long answer
    costs a handful of embeds rather than hundreds."""
    registered = 0
    text = ""
    for word in ANSWER.split():
        text += word + " "
        if E.remember_streaming("s1", text):
            registered += 1
    assert registered >= 1, "nothing was ever registered while streaming"
    assert registered <= len(ANSWER) // E._PARTIAL_STEP + 1, \
        f"re-embedded {registered} times — the growth gate is not holding"


def test_a_tiny_prefix_is_not_registered(stub_embedder):
    assert E.remember_streaming("s1", "short") is False


def test_the_growth_mark_resets_between_answers(stub_embedder):
    """Without the reset the NEXT answer inherits this one's length and its
    opening never registers — the same bug, one turn later."""
    E.remember_streaming("s1", ANSWER)
    E.reset_streaming("s1")
    # Comfortably past the growth step, so only the RESET can be what allows it.
    second = ("A second, completely different answer about thread pools and "
              "how they bound concurrency, long enough to clear the growth "
              "step on its own rather than by inheriting the previous "
              "answer's length.")
    assert len(second) > E._PARTIAL_STEP
    assert E.remember_streaming("s1", second) is True


def test_forgetting_a_session_clears_the_streaming_mark(stub_embedder):
    E.remember_streaming("s1", ANSWER)
    E.forget_session("s1")
    assert "s1" not in E._PARTIAL_AT


def test_registration_never_raises_without_an_embedder(monkeypatch):
    """Fail-open: the live path must survive a cold or missing embedder."""
    import app.rag.embedder as emb

    def boom(text):
        raise RuntimeError("embedder cold")

    monkeypatch.setattr(emb, "embed_one", boom)
    assert E.remember_streaming("s1", ANSWER) is True   # gate passed, embed failed
    assert E.is_candidate_echo("s1", ANSWER) == (False, 0.0)


# ── The wiring actually exists ──────────────────────────────────────────────

def test_the_live_path_registers_prefixes_while_streaming():
    """A perfect helper nothing calls fixes nothing. The token loop must invoke
    it, not just the completion handler."""
    import inspect

    from app.api import routes_ws
    src = inspect.getsource(routes_ws)
    assert "remember_streaming" in src, \
        "the live token loop never registers a streaming prefix"
    assert "reset_streaming" in src, \
        "the growth mark is never cleared at end of turn"
