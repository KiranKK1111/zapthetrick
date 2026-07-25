"""Stage-4 §3.3 — resume as structured JSON → ATS vs designed render."""
from __future__ import annotations

from app.documents import resume as R

_DATA = {
    "name": "Ada Lovelace",
    "contact": {"email": "ada@x.io", "location": "London",
                "links": ["github.com/ada"]},
    "summary": "Backend engineer.",
    "experience": [{"role": "SWE", "company": "Analytical Engines",
                    "dates": "2021–now", "bullets": ["Built the compiler",
                                                     "Cut latency 40%"]}],
    "skills": ["Python", "Go", "Kafka"],
    "projects": [{"name": "Notes", "detail": "A CLI", "bullets": ["Fast"]}],
    "education": [{"degree": "BSc CS", "school": "Cambridge", "dates": "2020"}],
}


class TestRender:
    def test_ats_is_single_column_no_tables(self):
        md = R.render_resume(_DATA, mode="ats")
        assert "# Ada Lovelace" in md
        assert "## Experience" in md and "## Skills" in md
        assert "Python, Go, Kafka" in md        # plain comma list, ATS-safe
        assert "|" not in md                    # no tables/columns
        assert "**" not in md                   # ATS stays plain

    def test_designed_is_richer_same_data(self):
        md = R.render_resume(_DATA, mode="designed")
        assert "**SWE — Analytical Engines**" in md   # bold role
        assert "**Skills:**" in md
        # Same content, both modes carry the bullets.
        assert "Built the compiler" in md

    def test_both_modes_from_same_data(self):
        ats = R.render_resume(_DATA, mode="ats")
        des = R.render_resume(_DATA, mode="designed")
        for token in ["Ada Lovelace", "Cut latency 40%", "Kafka", "Cambridge"]:
            assert token in ats and token in des

    def test_missing_fields_omitted(self):
        md = R.render_resume({"name": "Bob"})
        assert md.strip() == "# Bob"

    def test_never_raises(self):
        R.render_resume(None)                   # type: ignore[arg-type]
        R.render_resume({"experience": "bad"})


class TestPatch:
    def test_list_appends(self):
        out = R.patch_resume(_DATA, {"skills": ["Docker"]})
        assert out["skills"] == ["Python", "Go", "Kafka", "Docker"]
        assert _DATA["skills"] == ["Python", "Go", "Kafka"]   # original intact

    def test_scalar_replaces(self):
        out = R.patch_resume(_DATA, {"summary": "Staff engineer."})
        assert out["summary"] == "Staff engineer."

    def test_object_merges(self):
        out = R.patch_resume(_DATA, {"contact": {"phone": "555"}})
        assert out["contact"]["phone"] == "555"
        assert out["contact"]["email"] == "ada@x.io"   # kept
