"""Artifact patching (workspace-and-artifacts R6, task 6.2).

Pins Property 6: a clean targeted edit (add/replace/remove) updates in place; an
un-appliable instruction returns applied=False so the caller regenerates;
content identity is otherwise preserved.
"""
from __future__ import annotations

from app.artifacts.patch import apply_patch


def test_append_section():
    out, applied = apply_patch("# Doc\n\nIntro.", "add a section about Security")
    assert applied
    assert "Security" in out and out.startswith("# Doc")


def test_replace_text():
    out, applied = apply_patch("Use MySQL here.", "replace MySQL with PostgreSQL")
    assert applied and "PostgreSQL" in out and "MySQL" not in out


def test_remove_line():
    doc = "keep this\nremove the secret line\nkeep that"
    out, applied = apply_patch(doc, "remove the secret line")
    assert applied and "secret" not in out
    assert "keep this" in out and "keep that" in out


def test_replace_target_not_found_falls_back():
    out, applied = apply_patch("nothing here", "replace Redis with Memcached")
    assert applied is False and out == "nothing here"


def test_unrecognized_instruction_falls_back():
    out, applied = apply_patch("content", "make it more elegant somehow")
    assert applied is False and out == "content"


def test_empty_instruction_is_noop():
    out, applied = apply_patch("content", "")
    assert applied is False and out == "content"


def test_never_raises():
    out, applied = apply_patch(None, None)
    assert applied is False
