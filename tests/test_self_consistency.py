"""B3 — the candidate's own prior answers are surfaced as a consistency directive
so later answers don't contradict earlier ones."""
from app.live import memory as M


class _Turn:
    def __init__(self, q, a, t=""):
        self.question, self.answer, self.topic = q, a, t


class _Tracker:
    def __init__(self, turns):
        self._turns = turns


def _mem(turns):
    return M.for_tracker(_Tracker(turns))


def test_prior_answers_become_a_consistency_directive():
    m = _mem([
        _Turn("What messaging system did you use",
              "We used Apache Kafka for ordering guarantees."),
        _Turn("How did you scale it", "We partitioned by customer id."),
    ])
    d = M.consistency_directive(m)
    assert "consistent" in d.lower()
    assert "Kafka" in d and "partitioned" in d


def test_no_answers_yields_empty():
    assert M.consistency_directive(_mem([_Turn("q", "")])) == ""
    assert M.consistency_directive(_mem([])) == ""


def test_limits_to_n_recent_answers():
    turns = [_Turn(f"q{i}", f"answer number {i} with enough words here") for i in range(6)]
    d = M.consistency_directive(_mem(turns), n=2)
    # only the last two answered turns appear.
    assert d.count("- On ") == 2
    assert "answer number 5" in d and "answer number 4" in d


def test_never_raises_on_bad_turns():
    class Bad:
        _turns = [object()]          # no .answer / .question attrs
    assert M.consistency_directive(M.for_tracker(Bad())) == ""
