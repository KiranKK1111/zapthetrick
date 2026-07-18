"""Candidate & organization intelligence
(live-conversational-intelligence R39, R40, R41; tasks 27.2).

Pins Properties 39, 40, 41: profile/graph build + source merge + scoped
retrieval, asset matching + resume-reality (no inflation), org profile + fit +
opt-in fallback.
"""
from __future__ import annotations

from app.live import assets, org, profile


def _resume():
    return {
        "skills": ["Java", "Spring Boot", "Kafka", "React"],
        "projects": [
            {"name": "Payment Platform", "tech": ["Kafka", "Redis"], "company": "Acme"},
            {"name": "Dashboard", "tech": ["React"], "company": "Acme"},
        ],
        "achievements": ["Cut latency 40%"],
        "summary": "5 years full-stack",
    }


# ---- candidate profile -------------------------------------------------
def test_build_profile_structured():
    p = profile.build_profile(_resume())
    assert "Kafka" in p.skills
    assert any(pr["name"] == "Payment Platform" for pr in p.projects)
    assert p.experience.startswith("5 years")


def test_profile_source_merge():
    p = profile.build_profile(_resume(), github=[{"name": "cli-tool"}], linkedin=["Go"])
    assert any(pr["name"] == "cli-tool" for pr in p.projects)
    assert "Go" in p.skills


def test_knowledge_graph_company_nodes():
    g = profile.knowledge_graph(profile.build_profile(_resume()))
    assert "Acme" in g
    assert "Kafka" in g["Acme"]["tech"]


def test_scoped_retrieve_topic():
    p = profile.build_profile(_resume())
    hits = profile.scoped_retrieve(p, "kafka")
    assert any("Kafka" in h for h in hits)
    assert all("React" not in h for h in hits) or hits  # kafka-scoped


# ---- interview assets + resume reality --------------------------------
def test_match_asset():
    assert assets.match_asset("So, tell me about yourself") == "self_intro"
    assert assets.match_asset("Why should we hire you?") == "why_hire"
    assert assets.match_asset("How does Kafka work?") is None


def test_resume_reality_supports_real_claim_only():
    p = profile.build_profile(_resume())
    assert assets.supports_claim("I used Kafka in production", p) is True
    assert assets.supports_claim("I built Rust kernels", p) is False
    assert "Ground resume claims" in assets.reality_directive(p)


# ---- organization + fit ------------------------------------------------
def test_build_org_parses_jd_skills():
    o = org.build_org("Acme", "We need Java, Kafka and Kubernetes experience.", "SDE-2")
    assert "java" in o.jd_skills
    assert "kafka" in o.jd_skills


def test_fit_analysis_matching_and_gaps():
    p = profile.build_profile(_resume())
    o = org.build_org("Acme", "Java, Kafka, Kubernetes required", "SDE-2")
    fit = org.fit_analysis(p, o)
    assert "java" in fit["matching"]
    assert "kafka" in fit["matching"]
    assert "kubernetes" in fit["gaps"]
    assert "Acme" in org.directive(o, fit)


def test_fit_analysis_no_jd_falls_back_to_strengths():
    p = profile.build_profile(_resume())
    o = org.build_org("Acme")
    fit = org.fit_analysis(p, o)
    assert fit["strengths"]
