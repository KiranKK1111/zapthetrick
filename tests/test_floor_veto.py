"""The speaker-holds-floor veto: the interviewer handling their OWN business
("give me one moment", "let me tell you about the team") must never be answered.

Those utterances trip LITERAL cue/prefix lists ("give me", "describe", "tell
me"), which is why the veto is semantic. Fail-open by design: with no embedder
the veto is inert and the deterministic behaviour is unchanged, so these tests
skip rather than assert a false expectation."""
import pytest


def _embedder_ready() -> bool:
    try:
        from app.rag import embedder
        embedder.embed(["warm up"])
        return bool(embedder.is_ready())
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _embedder_ready(),
                                reason="semantic embedder unavailable")


FLOOR = [
    "give me one moment, my screen froze",
    "let me tell you a bit about the team",
    "let me pull up your resume for a second",
    "before we dive in, i want to describe the interview format",
    "i'll walk you through what we do here",
]

REQUESTS = [
    "walk me through your approach",
    "give me an example of that",
    "tell me about your current project",
    "describe a time you had a conflict with a teammate",
    "explain how kafka handles ordering",
]


@pytest.mark.parametrize("text", FLOOR)
def test_speaker_holding_floor_is_not_a_question(text):
    from app.question_detection.classifier import heuristic_classify
    from app.live.implicit import holds_floor
    assert holds_floor(text) is True, text
    assert heuristic_classify(text).is_question is False, text


@pytest.mark.parametrize("text", REQUESTS)
def test_real_requests_still_promote(text):
    """The veto must not swallow genuine requests — that would LOSE answers."""
    from app.question_detection.classifier import heuristic_classify
    from app.live.implicit import holds_floor
    assert holds_floor(text) is False, text
    assert heuristic_classify(text).is_question is True, text
