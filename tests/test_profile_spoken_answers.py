"""Live profile answers must be crisp, dictate-ready, first-person (user ask
2026-07-09 #6): the spoken profile prompt is selected for profile questions
and forbids essay formatting."""
from __future__ import annotations

from app.persona.voice import build_profile_answer_prompt
from app.tools.persona_answer import _build_messages

_PROFILE = {"summary": "Senior backend engineer",
            "skills": ["python", "kafka"]}


class TestSpokenProfilePrompt:
    def test_prompt_constraints(self):
        p = build_profile_answer_prompt(_PROFILE)
        assert "READ ALOUD" in p
        assert "First person" in p
        assert "NO headings" in p
        assert "NEVER invent" in p
        assert "Senior backend engineer" in p   # profile embedded

    def test_profile_q_selects_spoken_prompt(self):
        msgs = _build_messages(
            question="Tell me about yourself",
            profile=_PROFILE, context=None, prior_qa=None,
            qtype="behavioral", profile_q=True)
        assert "READ ALOUD" in msgs[0]["content"]

    def test_non_profile_q_keeps_detailed_prompt(self):
        msgs = _build_messages(
            question="What is Kafka?",
            profile=_PROFILE, context=None, prior_qa=None,
            qtype="technical_concept", profile_q=False)
        assert "READ ALOUD" not in msgs[0]["content"]

    def test_profile_q_without_profile_falls_back(self):
        # No resume uploaded → the no-resume guard directive handles it; the
        # spoken prompt (which asserts "FROM THE PROFILE ONLY") must not fire.
        msgs = _build_messages(
            question="Tell me about yourself",
            profile={}, context=None, prior_qa=None,
            qtype="behavioral", profile_q=True)
        assert "READ ALOUD" not in msgs[0]["content"]

    def test_directive_mentions_dictation(self):
        from app.live.profile import first_person_directive
        d = first_person_directive()
        assert "READ YOUR ANSWER ALOUD" in d
        assert "120-220 words" in d
