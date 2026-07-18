"""Domain-context builder for STT question repair (app.live.domain)."""
from __future__ import annotations

from app.live import domain as D


def test_vocab_from_skills_and_projects():
    prof = {"skills": ["Java", "Kubernetes", "Docker"],
            "projects": [{"name": "x", "tech": ["React", "Node"]}]}
    dc = D.build_domain(prof, None)
    vl = [v.lower() for v in dc.vocab]
    assert "kubernetes" in vl and "java" in vl and "react" in vl


def test_role_and_jd_terms():
    org = {"job_role": "SDE-2 Backend",
           "job_description": "Experience with Kafka, gRPC and Redis required."}
    dc = D.build_domain(None, org)
    assert dc.role == "SDE-2 Backend"
    vl = [v.lower() for v in dc.vocab]
    assert "kafka" in vl


def test_skills_as_comma_string():
    dc = D.build_domain({"skills": "Java, Spring Boot / PostgreSQL"}, None)
    vl = [v.lower() for v in dc.vocab]
    assert "java" in vl and "spring boot" in vl and "postgresql" in vl


def test_prompt_block_and_empty():
    dc = D.build_domain({"skills": ["Go", "gRPC"]}, {"job_role": "Backend"})
    blk = dc.prompt_block()
    assert "DOMAIN" in blk and "Backend" in blk and "gRPC" in blk
    # No context at all → empty block (cleaner prompt stays unchanged).
    empty = D.build_domain(None, None)
    assert empty.empty is True
    assert empty.prompt_block() == ""


def test_dedup_and_caps():
    dc = D.build_domain({"skills": ["Java", "java", "JAVA", "x"]}, None)  # dup + too-short
    vl = [v.lower() for v in dc.vocab]
    assert vl.count("java") == 1
    assert "x" not in dc.vocab            # length < 2 dropped


def test_never_raises_on_garbage():
    assert D.build_domain({"skills": 123, "projects": "nope"},
                          {"job_role": 5, "job_description": None}).prompt_block() == "" \
        or isinstance(D.build_domain({"skills": 123}, {}).prompt_block(), str)
