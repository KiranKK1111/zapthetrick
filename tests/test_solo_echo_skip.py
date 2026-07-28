"""Solo-mode candidate-echo skip (2026-07-28).

The echo skip used to be disabled in solo on the theory that content-matching
would suppress real questions from the single test voice. These tests pin the
property that makes enabling it safe: `is_candidate_echo` matches the utterance
against DISPLAYED ANSWER TEXT, so a read-aloud of the shown answer matches while
a genuine new question does not. Skips when the embedder is unavailable (the
matcher is embedding-based and fail-open)."""
import pytest

from app.live import echo


def _ready() -> bool:
    try:
        from app.rag import embedder
        embedder.embed(["warm"])
        return bool(embedder.is_ready())
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason="embedder unavailable")

_SID = "solo-echo-test"
_ANSWER = (
    "Kafka guarantees ordering within a partition by assigning each record a "
    "monotonically increasing offset; consumers read offsets sequentially, so "
    "per-key ordering is preserved by routing a key to one partition.")


@pytest.fixture(autouse=True)
def _store():
    echo.forget_session(_SID)
    echo.remember_answer(_SID, _ANSWER)
    yield
    echo.forget_session(_SID)


def test_reading_the_shown_answer_is_echo():
    is_echo, sim = echo.is_candidate_echo(
        _SID,
        "Kafka guarantees ordering within a partition by assigning each record "
        "a monotonically increasing offset so consumers read sequentially",
        0.72)
    assert is_echo is True and sim >= 0.72


def test_a_new_question_is_not_echo():
    # The solo tester's NEXT question must never be suppressed — this is the
    # property that makes enabling the skip in solo mode safe.
    for q in ("What is a consumer group?",
              "How would you scale a websocket service?",
              "Tell me about your current project"):
        is_echo, _ = echo.is_candidate_echo(_SID, q, 0.72)
        assert is_echo is False, q


def test_paraphrased_read_back_is_echo():
    is_echo, _ = echo.is_candidate_echo(
        _SID,
        "So basically Kafka keeps ordering inside a partition using increasing "
        "offsets and consumers just read them in sequence per key",
        0.72)
    assert is_echo is True
