"""Stage-6 §4.18 — Career Graph (source-span envelope) + JD tailoring + profile-Q."""
from __future__ import annotations

from app.live import career_graph as CG

_RESUME = ("Senior Backend Engineer at Acme. Built a Kafka pipeline that cut "
           "latency 40%. Skills: Python, Go, Kubernetes.")
_PROFILE = {
    "roles": ["Senior Backend Engineer"],
    "projects": [{"name": "Kafka pipeline"}],
    "metrics": ["cut latency 40%"],
    "skills": ["Python", "Go", "Rust"],   # Rust is NOT in the resume
}


class TestBuildGraph:
    def test_facts_extracted_by_kind(self):
        g = CG.build_career_graph(_PROFILE, _RESUME)
        assert [f.value for f in g.roles] == ["Senior Backend Engineer"]
        assert [f.value for f in g.projects] == ["Kafka pipeline"]
        assert "Python" in [f.value for f in g.skills]

    def test_source_spans_attached_for_supported_facts(self):
        g = CG.build_career_graph(_PROFILE, _RESUME)
        role = g.roles[0]
        assert role.grounded is True
        # The span points at the real resume text.
        assert _RESUME[role.source.start:role.source.end].lower() \
            == "senior backend engineer"

    def test_unsupported_fact_is_ungrounded(self):
        # "Rust" is in the profile but not the resume → the envelope flags it.
        g = CG.build_career_graph(_PROFILE, _RESUME)
        assert "Rust" in [f.value for f in g.ungrounded_facts()]
        assert "Python" not in [f.value for f in g.ungrounded_facts()]

    def test_no_resume_text_leaves_all_ungrounded(self):
        g = CG.build_career_graph(_PROFILE, "")
        assert g.grounded_facts() == []

    def test_never_raises_on_bad_input(self):
        assert CG.build_career_graph(None, None).all_facts() == []  # type: ignore[arg-type]

    def test_as_dict_shape(self):
        d = CG.build_career_graph(_PROFILE, _RESUME).as_dict()
        assert set(d) == {"roles", "projects", "metrics", "skills"}
        assert d["skills"][0]["grounded"] in (True, False)


class TestTailoring:
    def test_emphasizes_matching_grounded_facts(self):
        g = CG.build_career_graph(_PROFILE, _RESUME)
        t = CG.tailor_to_jd(g, ["Kafka", "Python"])
        assert "Kafka pipeline" in t.emphasize
        assert "Python" in t.emphasize

    def test_flags_jd_gaps(self):
        g = CG.build_career_graph(_PROFILE, _RESUME)
        t = CG.tailor_to_jd(g, ["Terraform", "Kafka"])
        assert "terraform" in t.gaps
        assert "Kafka pipeline" in t.emphasize

    def test_ungrounded_fact_does_not_satisfy_jd(self):
        # Rust is ungrounded → a Rust JD requirement is still a GAP.
        g = CG.build_career_graph(_PROFILE, _RESUME)
        t = CG.tailor_to_jd(g, ["Rust"])
        assert "rust" in t.gaps and t.emphasize == []

    def test_empty_jd_is_empty_tailoring(self):
        g = CG.build_career_graph(_PROFILE, _RESUME)
        t = CG.tailor_to_jd(g, [])
        assert t.emphasize == [] and t.gaps == []


class TestProfileQuestion:
    def test_profile_questions_detected(self):
        assert CG.is_profile_question("tell me about yourself") is True
        assert CG.is_profile_question("walk me through your projects") is True

    def test_non_profile_not_detected(self):
        assert CG.is_profile_question("what does our company do") is False
        assert CG.is_profile_question("explain how kafka works") is False

    def test_empty_is_false(self):
        assert CG.is_profile_question("") is False


class TestFlag:
    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "profile_library", False, raising=False)
        assert CG.enabled() is False

    def test_enabled_reads_flag(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "profile_library", True, raising=False)
        assert CG.enabled() is True
