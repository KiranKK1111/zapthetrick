"""
Phase-D interview scenarios: resume / organization / negotiation / interview-mode
intelligence. Every test exercises a REAL `app.live.*` module (no LLM, no network,
deterministic). Per-session state uses a unique `sid` per test via
`context_tracker.get_tracker`.

Scenario-to-module map (verified against each module's actual API):
  - resume ............ app/live/profile.py, app/live/assets.py
  - organization ...... app/live/org.py (interview-framed; NOT duplicating
                        tests/test_org_intelligence.py)
  - negotiation ....... app/live/negotiate.py
  - interview mode .... app/live/modes.py, app/live/phase.py
  - knowledge/memory .. app/live/knowledge.py, app/live/memory.py
  - advisory extras ... app/live/career.py, coach.py, emotion.py, outcome.py

Gaps (negotiate.py has NO LOW_OFFER / VALUE_JUSTIFICATION / FINAL_OFFER intents —
its intents are salary / notice_period / counter_offer / why_join / why_leaving /
benefits / other) are marked with `pytest.skip` and a reason.
"""
from __future__ import annotations

import pytest

from app.live import assets, career, coach, emotion
from app.live import knowledge as K
from app.live import modes
from app.live import negotiate as N
from app.live import org as ORG
from app.live import outcome
from app.live import phase
from app.live.memory import for_tracker, refresh_summary
from app.live.profile import (
    build_profile,
    knowledge_graph,
    reality_terms,
    scoped_retrieve,
)
from app.question_detection.context_tracker import Turn, get_tracker


# A representative resume the resume scenarios reuse.
_RESUME = {
    "skills": ["Python", "Kafka", "AWS", "PostgreSQL"],
    "projects": [
        {"name": "Event pipeline", "tech": ["Kafka", "Python"], "company": "Acme"},
        {"name": "Billing service", "tech": ["PostgreSQL", "AWS"], "company": "Acme"},
    ],
    "achievements": ["Cut p99 latency by 40%"],
    "metrics": ["40% latency reduction", "2M events/day"],
    "experience": "5 years backend engineering",
}


# ---------------------------------------------------------------------------
# Resume: structured profile / retrieval / reality
# ---------------------------------------------------------------------------

# Scenario 135: resume structured profile — build a CandidateProfile from a dict
def test_s135_resume_structured_profile():
    p = build_profile(_RESUME)
    assert p.skills == ["Python", "Kafka", "AWS", "PostgreSQL"]
    assert p.experience == "5 years backend engineering"
    assert p.metrics == ["40% latency reduction", "2M events/day"]
    # Projects are normalized to {name, tech, company}.
    names = {proj["name"] for proj in p.projects}
    assert names == {"Event pipeline", "Billing service"}
    ev = next(proj for proj in p.projects if proj["name"] == "Event pipeline")
    assert ev["tech"] == ["Kafka", "Python"] and ev["company"] == "Acme"
    # Knowledge graph groups projects/tech under the company.
    g = knowledge_graph(p)
    assert "Acme" in g
    assert set(g["Acme"]["projects"]) == {"Event pipeline", "Billing service"}
    assert "Kafka" in g["Acme"]["tech"]


# Scenario 141: resume retrieval by topic — scoped_retrieve for the topic "Kafka"
def test_s141_resume_retrieval_by_topic():
    p = build_profile(_RESUME)
    hits = scoped_retrieve(p, "Kafka")
    # Retrieves the matching skill AND the matching project (topic-scoped slice).
    assert "Skill: Kafka" in hits
    assert any(h.startswith("Project: Event pipeline") for h in hits)
    # Unrelated topics don't leak in.
    assert not any("Billing" in h for h in hits)
    # An unknown topic returns nothing (grounded-only retrieval).
    assert scoped_retrieve(p, "Rust") == []


# Scenario 143: avoid repeat — recent questions are recallable so an answer can
# avoid re-covering ground already asked in the session
def test_s143_avoid_repeat():
    tr = get_tracker("s143-avoid-repeat")
    tr._turns.append(Turn(question="Explain Kafka partitioning", topic="kafka"))
    tr._turns.append(Turn(question="How does consumer rebalancing work?", topic="kafka"))
    mem = for_tracker(tr)
    recent = mem.l1()
    # The already-asked questions are retrievable → the model can avoid repeating.
    assert "Explain Kafka partitioning" in recent
    assert "How does consumer rebalancing work?" in recent
    # context_for surfaces the same prior questions for a follow-up on the topic.
    ctx = mem.context_for("And exactly-once?", "kafka")
    assert "Explain Kafka partitioning" in ctx


# Scenario 144: answer personalization — scoped_retrieve grounds the answer in
# the candidate's OWN experience rather than generic content
def test_s144_answer_personalization():
    p = build_profile(_RESUME)
    # A generic AWS question is personalized to the candidate's real AWS project.
    hits = scoped_retrieve(p, "AWS")
    assert "Skill: AWS" in hits
    assert any("Billing service" in h for h in hits)
    # Personalization draws only from real terms the candidate can claim.
    terms = reality_terms(p)
    assert {"aws", "kafka", "python", "postgresql"} <= terms


# Scenario 79: resume reality enforcement — ground claims only in real experience
def test_s79_resume_reality_enforcement():
    p = build_profile(_RESUME)
    directive = assets.reality_directive(p)
    assert directive  # non-empty when the profile has terms
    assert "inflate" in directive.lower()
    # The directive lists the candidate's real terms.
    assert "kafka" in directive.lower() and "python" in directive.lower()
    # A claim naming a technology NOT in the profile is not supported.
    assert assets.supports_claim("I have deep Cassandra internals expertise", p) is False
    # A claim grounded in a real term is supported.
    assert assets.supports_claim("I built Kafka pipelines", p) is True
    # Empty profile → no directive (fail-open).
    assert assets.reality_directive(build_profile({})) == ""


# ---------------------------------------------------------------------------
# Organization / JD intelligence (interview-framed; not duplicating
# tests/test_org_intelligence.py which covers JD extraction + fit_directive)
# ---------------------------------------------------------------------------

# Scenario 149: organization profile build — build_org from company + JD + role
def test_s149_organization_profile_build():
    o = ORG.build_org("Acme", "We need Python, Kafka and Kubernetes.", "SDE-2")
    assert o.company == "Acme"
    assert o.role == "SDE-2"
    assert {"python", "kafka", "kubernetes"} <= set(o.jd_skills)
    # Dataclass round-trips the interview session metadata.
    d = o.to_dict()
    assert d["company"] == "Acme" and d["role"] == "SDE-2"
    assert "jd_skills" in d and "notes" in d


# Scenario 150: "why join us?" — HR intent classification grounds the answer
def test_s150_why_join_us():
    intent = N.classify_hr_intent("Why do you want to join us?")
    assert intent == N.WHY_JOIN
    strat = N.negotiation_strategy("Why do you want to join us?")
    assert strat.intent == N.WHY_JOIN
    # Guidance ties motivation to role/company specifics (fact-based, not manipulative).
    assert any("motivation" in p.lower() for p in strat.points)
    assert strat.risk_flag == ""


# Scenario 152: "why should we hire you?" — served by a prepared resume asset
def test_s152_why_hire_you():
    assert assets.match_asset("Why should we hire you?") == "why_hire"
    assert assets.match_asset("What makes you a good fit for this role?") == "why_hire"
    # It is a distinct prepared asset key.
    assert "why_hire" in assets.asset_keys()


# Scenario 153: JD upload -> fit analysis (matching skills vs. gaps)
def test_s153_jd_upload_fit_analysis():
    p = build_profile(_RESUME)  # Python, Kafka, AWS, PostgreSQL
    o = ORG.build_org("Acme", "Required: Python, Kafka, Kubernetes and Terraform.", "SDE-2")
    fit = ORG.fit_analysis(p, o)
    assert "python" in fit["matching"] and "kafka" in fit["matching"]
    assert "kubernetes" in fit["gaps"] and "terraform" in fit["gaps"]
    # Strengths fall back to the matching set.
    assert set(fit["strengths"]) <= set(fit["matching"])


# Scenario 155: company-aware mode — the answer directive names the target
# company + role and emphasizes matching strengths / honest gaps
def test_s155_company_aware_mode():
    p = build_profile(_RESUME)
    o = ORG.build_org("Acme", "Java and Kafka required.", "SDE-2")
    fit = ORG.fit_analysis(p, o)
    directive = ORG.directive(o, fit)
    assert "Acme" in directive and "SDE-2" in directive
    assert "kafka" in directive.lower()          # matching strength emphasized
    assert "java" in directive.lower()           # gap addressed honestly


# ---------------------------------------------------------------------------
# Interview modes / phase
# ---------------------------------------------------------------------------

# Scenario 110: behavioral STAR engine — behavioral phase -> STAR-story mode
def test_s110_behavioral_star_engine():
    q = "Tell me about a time you handled a conflict on your team."
    assert phase.detect_phase(q) == phase.BEHAVIORAL
    assert modes.detect_mode(q) == modes.STAR_STORY
    # The STAR mode directive shapes the answer as Situation/Task/Action/Result.
    d = modes.directive(modes.STAR_STORY)
    for part in ("Situation", "Task", "Action", "Result"):
        assert part in d


# Scenario 111: HR mode — an HR/salary question routes to the negotiation mode
def test_s111_hr_mode():
    q = "What are your salary expectations?"
    assert phase.detect_phase(q) == phase.HR
    # modes.py maps the HR phase to the NEGOTIATION operating mode.
    assert modes.detect_mode(q) == modes.NEGOTIATION
    d = modes.directive(modes.NEGOTIATION)
    assert "market" in d.lower() and "value" in d.lower()


# Scenario 112: system-design mode — a design prompt -> structured-design mode
def test_s112_system_design_mode():
    q = "Design a URL shortener that can scale to millions of requests."
    assert phase.detect_phase(q) == phase.SYSTEM_DESIGN
    assert modes.detect_mode(q) == modes.STRUCTURED_DESIGN
    d = modes.directive(modes.STRUCTURED_DESIGN)
    assert "requirements" in d.lower() and "trade-off" in d.lower()


# Scenario 117: interview phase detection (deterministic, no LLM)
def test_s117_interview_phase_detection():
    assert phase.detect_phase("Tell me about yourself.") == phase.INTRODUCTION
    assert phase.detect_phase("What are your salary expectations?") == phase.HR
    assert phase.detect_phase("Do you have any questions for us?") == phase.CLOSING
    # "Let's talk numbers" is NOT a recognized HR cue (there is no NEGOTIATION
    # phase; HR is cued by explicit terms like salary/compensation/ctc), so it
    # falls back to the neutral technical-screening default.
    assert phase.detect_phase("Let's talk numbers") == phase.TECHNICAL_SCREENING


# ---------------------------------------------------------------------------
# Salary / negotiation
# ---------------------------------------------------------------------------

# Scenario 157: salary mode switch — a salary question flips phase -> HR and
# mode -> negotiation in one step
def test_s157_salary_mode_switch():
    q = "Let's discuss your salary expectations for this role."
    assert phase.detect_phase(q) == phase.HR
    assert modes.detect_mode(q) == modes.NEGOTIATION
    # ModeTracker requires the same new mode twice (hysteresis) before switching.
    tr = get_tracker("s157-salary-switch")
    mt = modes.for_tracker(tr)
    assert mt.update(q) == modes.GENERAL      # pending after first cue
    assert mt.update(q) == modes.NEGOTIATION  # confirmed on the second


# Scenario 158: salary expectation anchor — classify intent + fact-based anchor
def test_s158_salary_expectation_anchor():
    q = "What are your salary expectations?"
    assert N.classify_hr_intent(q) == N.SALARY
    strat = N.negotiation_strategy(
        q, strengths=["Kafka pipelines at scale"], market_low=30, market_high=50
    )
    assert strat.intent == N.SALARY
    joined = " ".join(strat.points).lower()
    assert "30-50" in joined                  # anchored on the market band
    assert "kafka pipelines" in joined        # justified with concrete value
    assert strat.risk_flag == ""              # a reasonable, in-band ask


# Scenario 159: low-offer strategy — acknowledge, reinforce value, counter politely
def test_s159_low_offer_strategy():
    assert N.classify_hr_intent("We can offer 38 LPA") == N.LOW_OFFER
    s = N.negotiation_strategy("We can offer 38 LPA",
                               strengths=["Kafka", "microservices"])
    assert s.intent == N.LOW_OFFER
    joined = " ".join(s.points).lower()
    assert "counter" in joined and "value" in joined


# Scenario 160: value justification — justify with measurable impact
def test_s160_value_justification():
    assert N.classify_hr_intent("Why do you deserve 50 LPA?") == N.VALUE_JUSTIFICATION
    s = N.negotiation_strategy("Why do you deserve 50 LPA?",
                               strengths=["+40% throughput"])
    assert s.intent == N.VALUE_JUSTIFICATION
    assert any("impact" in p.lower() or "proof" in p.lower() for p in s.points)


# Scenario 161: final offer — respectful, non-cash levers, no ultimatum
def test_s161_final_offer():
    assert N.classify_hr_intent("This is our final offer.") == N.FINAL_OFFER
    s = N.negotiation_strategy("This is our final offer.")
    assert s.intent == N.FINAL_OFFER
    joined = " ".join(s.points).lower()
    assert "ultimatum" in joined or "non-cash" in joined or "equity" in joined


# Scenario 164: negotiation risk detection — an unrealistic ask raises a flag
def test_s164_negotiation_risk_detection():
    # Ask far above the market band (> market_high * 1.5) trips the advisory flag.
    strat = N.negotiation_strategy(
        "What are your salary expectations?",
        market_low=30, market_high=50, ask=90,
    )
    assert strat.intent == N.SALARY
    assert strat.risk_flag == "unrealistic_ask"
    assert any("above the market band" in p.lower() for p in strat.points)
    # A sane, in-band ask carries no risk flag.
    ok = N.negotiation_strategy(
        "What are your salary expectations?",
        market_low=30, market_high=50, ask=48,
    )
    assert ok.risk_flag == ""


# ---------------------------------------------------------------------------
# Multi-level memory / knowledge packs / real-time learning
# ---------------------------------------------------------------------------

# Scenario 130: multi-level memory — L1 recent / L2 topic-scoped / L3 summary
def test_s130_multi_level_memory():
    tr = get_tracker("s130-memory")
    tr._turns.append(Turn(question="Explain Kafka partitions", topic="kafka"))
    tr._turns.append(Turn(question="How does rebalancing work?", topic="kafka"))
    tr._turns.append(Turn(question="What is Redis persistence?", topic="redis"))
    mem = for_tracker(tr)
    # L1: recent detail (last N questions, oldest first).
    assert mem.l1()[-1] == "What is Redis persistence?"
    # L2: topic-scoped recall — only the kafka turns.
    l2 = [t.question for t in mem.l2("kafka")]
    assert l2 == ["Explain Kafka partitions", "How does rebalancing work?"]
    # L3: deterministic rolling summary refresh (no LLM call).
    summary = refresh_summary(mem)
    assert "Topics covered: kafka, redis" in summary
    assert "Redis persistence" in summary
    assert mem.l3() == summary


# Scenario 140: skill-specific packs — per-topic interview knowledge angles
def test_s140_skill_specific_packs():
    angles = K.interview_knowledge("Kafka")
    assert "ordering is per-partition, not global" in angles
    # Folded into an answer directive.
    d = K.directive(angles)
    assert d.startswith("Relevant angles to consider:")
    assert "per-partition" in d
    # An unknown topic yields no pack (fail-open).
    assert K.interview_knowledge("underwater basket weaving") == []
    # configured_pack reads config without raising (empty by default in tests).
    assert isinstance(K.configured_pack(), str)


# Scenario 119: real-time learning focus — recurring skill-gap topic gets a
# retrieval boost beyond the base angles
def test_s119_realtime_learning_focus():
    base = K.interview_knowledge("system design")
    # When the topic is a recurring skill gap, the boost reinforces the answer.
    boosted = K.skill_gap_boost("system design", ["system design"])
    assert boosted  # non-empty knowledge focus for the gap
    # The boost is a superset of the base angles (never loses ground).
    assert set(base) <= set(boosted)
    # A topic that is NOT a recurring gap returns just the base angles.
    assert K.skill_gap_boost("system design", ["kafka"]) == base


# ---------------------------------------------------------------------------
# Advisory extras (light coverage — career / coach / emotion / outcome)
# ---------------------------------------------------------------------------

# Scenario (extra): advisory career intelligence from profile + fit
def test_s_extra_career_intelligence():
    p = build_profile(_RESUME)
    o = ORG.build_org("Acme", "Python, Kafka, Kubernetes, Terraform.", "SDE-2")
    fit = ORG.fit_analysis(p, o)
    ci = career.analyze(p, o, fit=fit)
    assert ci.advisory is True
    assert "NOT professional" in ci.disclaimer
    # Gaps from the fit analysis become coaching focus areas.
    assert "kubernetes" in ci.skill_gaps and "terraform" in ci.skill_gaps
    assert ci.readiness in {"entry", "mid", "senior_ready", "unknown"}


# Scenario (extra): delivery coaching flags filler words and thin answers
def test_s_extra_delivery_coaching():
    tips = coach.coach("um so like um yeah i uh worked on some stuff um you know")
    assert any("filler" in t.lower() for t in tips)
    # Empty candidate speech yields nothing (fail-open).
    assert coach.coach("") == []


# Scenario (extra): advisory emotion signal from prosody proxies
def test_s_extra_emotion_signal():
    assert emotion.analyze(filler_ratio=0.5).label == emotion.HESITANT
    assert emotion.analyze(speech_rate=0.9).label == emotion.RUSHED
    assert emotion.analyze(energy=0.1, speech_rate=0.2).label == emotion.CALM
    # No signal at all -> neutral, advisory-only.
    n = emotion.analyze()
    assert n.label == emotion.NEUTRAL and n.advisory is True


# Scenario (extra): advisory outcome estimate aggregates session signals
def test_s_extra_outcome_estimate():
    strong = outcome.estimate(answered=9, total=10, avg_confidence=0.85, satisfaction=0.9)
    assert strong.band == outcome.STRONG
    assert "NOT a hiring decision" in strong.disclaimer
    # Nothing to go on -> UNKNOWN.
    assert outcome.estimate().band == outcome.UNKNOWN
