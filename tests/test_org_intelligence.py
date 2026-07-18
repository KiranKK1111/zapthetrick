"""Org intelligence: JD skill extraction, fit directive, and live-session
metadata persistence (org_name / job_role / job_description / notes)."""
from __future__ import annotations

import asyncio

import app.live.org as org
from app.live.profile import CandidateProfile


# --------------------------------------------------------------------------
# JD skill extraction
# --------------------------------------------------------------------------

def test_jd_skill_extraction_finds_lexicon_skills():
    jd = (
        "We are hiring a Senior Backend Engineer. Must have Java, Spring Boot, "
        "Kafka and Kubernetes; PostgreSQL and Redis are a plus. You will build "
        "microservices with REST and GraphQL, and own CI/CD with Docker on AWS."
    )
    o = org.build_org("Acme", jd, "Senior Backend Engineer")
    for skill in ("java", "spring boot", "kafka", "kubernetes", "postgresql",
                  "redis", "microservices", "rest", "graphql", "ci/cd",
                  "docker", "aws"):
        assert skill in o.jd_skills, skill
    # No false positives from unrelated lexicon entries.
    assert "javascript" not in o.jd_skills
    assert "angular" not in o.jd_skills


def test_jd_extraction_symbol_skills_and_quoted_terms():
    jd = 'Strong C++ required. Experience with "Temporal" workflows is a bonus.'
    o = org.build_org("Acme", jd)
    assert "c++" in o.jd_skills
    assert "temporal" in o.jd_skills          # quoted-term heuristic


def test_jd_extraction_camelcase_and_acronym_heuristics():
    jd = "You will use TypeScript and GraphQL. GCP experience preferred. EEO employer."
    o = org.build_org("Acme", jd)
    assert "typescript" in o.jd_skills
    assert "graphql" in o.jd_skills
    assert "gcp" in o.jd_skills
    assert "eeo" not in o.jd_skills           # boilerplate acronym filtered


def test_jd_extraction_go_needs_cased_form():
    o = org.build_org("Acme", "Candidates should be ready to go fast and iterate.")
    assert "go" not in o.jd_skills
    o2 = org.build_org("Acme", "We write services in Go and Rust.")
    assert "go" in o2.jd_skills


# --------------------------------------------------------------------------
# build_org signature (backward compatible + new keyword-only params)
# --------------------------------------------------------------------------

def test_build_org_positional_compat_and_new_kwargs():
    # Existing positional call style keeps working.
    o = org.build_org("Acme", "We need python and kubernetes.", "Backend")
    assert o.company == "Acme" and o.role == "Backend"
    assert "python" in o.jd_skills and "kubernetes" in o.jd_skills
    # New keyword-only params.
    o2 = org.build_org("Acme", job_role="Platform Engineer", notes="Series B fintech")
    assert o2.role == "Platform Engineer"
    assert o2.notes == "Series B fintech"


# --------------------------------------------------------------------------
# fit directive
# --------------------------------------------------------------------------

def test_fit_directive_mentions_matching_and_gap_skills():
    prof = CandidateProfile(skills=["Java", "Kafka"])
    o = org.build_org("Acme", "Java, Kafka and Kubernetes required.", "SDE-2")
    d = org.fit_directive(o, prof)
    assert "Acme" in d and "SDE-2" in d
    assert "java" in d and "kafka" in d        # matching skills
    assert "kubernetes" in d                   # gap skill
    assert "Emphasize" in d and "address" in d # one-line guidance
    assert len(d) < 500


def test_fit_directive_empty_jd_degrades_gracefully():
    o = org.build_org("Acme")                  # no JD, no role, no notes
    d = org.fit_directive(o, CandidateProfile())
    assert "Acme" in d                         # still mentions the org
    assert len(d) < 500
    # And a fully-empty org never raises.
    assert org.fit_directive(org.build_org(), None) == ""


def test_fit_directive_includes_notes_and_stays_compact():
    prof = CandidateProfile(skills=["python", "aws", "docker", "kafka"])
    jd = ("Python, AWS, Docker, Kafka, Kubernetes, Terraform, Postgres, Redis, "
          "GraphQL, gRPC, Spark, Airflow, Snowflake, dbt, Jenkins and Helm.")
    o = org.build_org("Globex", jd, "Staff Engineer",
                      notes="They value ownership and pragmatism " * 8)
    d = org.fit_directive(o, prof)
    assert "Globex" in d and "Notes:" in d
    assert len(d) < 500


# --------------------------------------------------------------------------
# create-session route persists the new metadata (no DB — fake repo/session)
# --------------------------------------------------------------------------

def test_create_live_session_persists_new_metadata(monkeypatch):
    import storage.repos as repos
    from app.api import routes_live

    captured: dict = {}

    class _FakeRow:
        def __init__(self, **kw):
            self.id = "11111111-1111-1111-1111-111111111111"
            self.title = kw.get("title")
            self.session_metadata = kw.get("session_metadata")
            self.resume_id = None
            self.started_at = None
            self.updated_at = None
            self.message_count = 0
            self.last_message_at = None

    class _FakeRepo:
        def __init__(self, session):
            pass

        async def create(self, **kw):
            captured.update(kw)
            return _FakeRow(**kw)

    class _FakeSession:
        async def commit(self):
            pass

    monkeypatch.setattr(repos, "SessionRepo", _FakeRepo)

    body = routes_live.CreateLiveSession(
        org_name="Acme",
        job_role="SDE-2",
        job_description="Java and Kafka required.",
        notes="Onsite loop, 4 rounds.",
    )
    out = asyncio.run(routes_live.create_live_session(body, session=_FakeSession()))

    md = captured["session_metadata"]
    assert md["org_name"] == "Acme"
    assert md["job_role"] == "SDE-2"
    assert md["job_description"] == "Java and Kafka required."
    assert md["notes"] == "Onsite loop, 4 rounds."
    # Response stays backward compatible and ADDS the new keys.
    assert out["org_name"] == "Acme"
    assert out["job_role"] == "SDE-2"
    assert out["job_description"] == "Java and Kafka required."
    assert out["notes"] == "Onsite loop, 4 rounds."


def test_create_live_session_defaults_are_empty(monkeypatch):
    import storage.repos as repos
    from app.api import routes_live

    captured: dict = {}

    class _FakeRow:
        def __init__(self, **kw):
            self.id = "22222222-2222-2222-2222-222222222222"
            self.title = kw.get("title")
            self.session_metadata = kw.get("session_metadata")
            self.resume_id = None
            self.started_at = None
            self.updated_at = None
            self.message_count = 0
            self.last_message_at = None

    class _FakeRepo:
        def __init__(self, session):
            pass

        async def create(self, **kw):
            captured.update(kw)
            return _FakeRow(**kw)

    class _FakeSession:
        async def commit(self):
            pass

    monkeypatch.setattr(repos, "SessionRepo", _FakeRepo)

    body = routes_live.CreateLiveSession()   # old callers send only org_name (or nothing)
    out = asyncio.run(routes_live.create_live_session(body, session=_FakeSession()))

    md = captured["session_metadata"]
    assert md == {"org_name": "Interview", "job_role": "",
                  "job_description": "", "notes": "", "experience_level": ""}
    assert out["job_role"] == "" and out["notes"] == ""
