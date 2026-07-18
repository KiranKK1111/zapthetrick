"""
Dual-source continuity tests — hear both voices, act on one
(live-conversational-intelligence dual-source enhancement).

Covers the four pieces:
  1. role tag threading (candidate absorbed, never answered)
  2. role-aware shared conversation graph (interviewer / candidate / assistant)
  3. commitments store (stated salary/offer/notice + interviewer pushback signal)
  4. negotiation reads the candidate's stated figure + interviewer signal
"""
from __future__ import annotations

from app.live import conversation as _conv
from app.live import negotiate as _negot
from app.live import world_model as _wm


# ---- Piece 2: role-aware shared conversation graph -------------------------

def test_conversation_log_role_tagged_context():
    log = _conv.ConversationLog()
    log.add(_conv.INTERVIEWER, "What's your salary expectation?", topic="salary")
    log.add(_conv.CANDIDATE, "I'm looking for around 25 LPA.", topic="salary")
    log.add(_conv.ASSISTANT, "Anchor on the market band and justify with impact.", topic="salary")
    lines = log.context_lines("salary")
    joined = "\n".join(lines)
    assert "Interviewer:" in joined
    assert "You:" in joined            # candidate rendered as "You"
    assert "Assistant suggested:" in joined
    assert log.last_candidate() == "I'm looking for around 25 LPA."


def test_conversation_log_unknown_role_defaults_interviewer_and_bounded():
    log = _conv.ConversationLog(maxlen=3)
    log.add("mystery", "x")
    log.add(_conv.CANDIDATE, "a")
    log.add(_conv.CANDIDATE, "b")
    log.add(_conv.CANDIDATE, "c")   # evicts the first
    entries = log.entries()
    assert len(entries) == 3
    assert entries[0].role == _conv.CANDIDATE  # the "mystery" one was evicted


def test_conversation_for_tracker_persists():
    class T:
        pass
    t = T()
    assert _conv.for_tracker(t) is _conv.for_tracker(t)


# ---- Piece 3: commitments store + extraction -------------------------------

def test_extract_candidate_salary_offer_notice():
    c = _wm.extract_commitments("I'm looking for around 25 LPA", role="candidate")
    assert "salary" in c and "25" in c["salary"]
    c2 = _wm.extract_commitments("I have another offer on the table", role="candidate")
    assert c2.get("competing_offer") == "yes"
    c3 = _wm.extract_commitments("my notice period is 30 days", role="candidate")
    assert c3.get("notice_period") == "30 days"


def test_extract_interviewer_pushback_signal():
    s = _wm.extract_commitments("That's a bit high for this band", role="interviewer")
    assert s.get("salary_signal") == "interviewer_thinks_high"
    s2 = _wm.extract_commitments("Honestly you could ask for more", role="interviewer")
    assert s2.get("salary_signal") == "interviewer_thinks_low"


def test_record_and_read_commitments():
    m = _wm.InterviewWorldModel(topic="salary")
    _wm.record_commitment(m, "salary", "candidate", "25 LPA", topic="salary")
    _wm.record_commitment(m, "salary_signal", "interviewer", "interviewer_thinks_high",
                          topic="salary")
    comms = _wm.commitments_for(m, "salary")
    assert comms["salary"]["role"] == "candidate"
    assert comms["salary"]["value"] == "25 LPA"
    assert comms["salary_signal"]["role"] == "interviewer"
    # Later statement supersedes.
    _wm.record_commitment(m, "salary", "candidate", "28 LPA", topic="salary")
    assert _wm.commitments_for(m, "salary")["salary"]["value"] == "28 LPA"


# ---- Piece 4: negotiation reacts to what was said --------------------------

def test_negotiation_holds_candidate_stated_figure():
    strat = _negot.negotiation_strategy(
        "Can you be flexible on compensation?",
        strengths=["python"],
        candidate_stated={"salary": "25 LPA"},
        interviewer_signal="interviewer_thinks_high",
    )
    joined = " ".join(strat.points).lower()
    assert "25 lpa" in joined                      # references the stated figure
    assert "hold" in joined or "justify" in joined  # hold, don't undercut
    assert "high" in joined                         # reacts to the pushback signal


def test_negotiation_generic_when_nothing_stated():
    strat = _negot.negotiation_strategy("What's your salary expectation?")
    joined = " ".join(strat.points).lower()
    # No stated figure → the generic "state a range" guidance, no fabricated number.
    assert "range" in joined
    assert "25" not in joined


def test_negotiation_competing_offer_from_commitments():
    strat = _negot.negotiation_strategy(
        "Do you have another offer?",
        candidate_stated={"competing_offer": "yes"},
    )
    joined = " ".join(strat.points).lower()
    assert "competing" in joined or "offer" in joined
    # Never manipulative even with an offer on record.
    assert "bluff" not in joined and "fake" not in joined


def test_negotiation_still_no_manipulation():
    strat = _negot.negotiation_strategy(
        "What's your expected salary?",
        candidate_stated={"salary": "25 LPA"},
    )
    joined = " ".join(strat.points).lower()
    for bad in ("lie", "bluff", "fabricate", "threaten"):
        assert bad not in joined
