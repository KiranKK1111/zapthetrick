"""Tests for the diff-edits engine (vNext §8.8, Stage 8 Component A)."""
from __future__ import annotations

import asyncio

import app.documents.diff_edits as D

_DOC = "# Report\n\nStatus: draft.\n\nThe color is blue.\nThe color is blue.\n"


class _Res:
    def __init__(self, obj):
        self.obj = obj


def _run(coro):
    return asyncio.run(coro)


# ---- apply_patches: happy path -------------------------------------------
def test_unique_replace_applies():
    r = D.apply_patches(_DOC, [D.Patch("Status: draft.", "Status: final.")])
    assert r.ok and r.applied == 1 and r.changed
    assert "Status: final." in r.content
    assert "draft" not in r.content


def test_replace_all_replaces_every_occurrence():
    r = D.apply_patches(_DOC, [D.Patch("The color is blue.", "The color is red.",
                                       replace_all=True)])
    assert r.ok and r.applied == 1
    assert r.content.count("red") == 2
    assert "blue" not in r.content


def test_dict_patches_are_coerced():
    r = D.apply_patches(_DOC, [{"old_str": "Status: draft.",
                                "new_str": "Status: shipped."}])
    assert r.ok and "shipped" in r.content


def test_sequential_patches_apply_in_order():
    r = D.apply_patches(_DOC, [
        D.Patch("Status: draft.", "Status: final."),
        D.Patch("# Report", "# Final Report"),
    ])
    assert r.ok and r.applied == 2
    assert "# Final Report" in r.content and "Status: final." in r.content


# ---- apply_patches: rejection is atomic ----------------------------------
def test_ambiguous_match_rejected_without_mutation():
    r = D.apply_patches(_DOC, [D.Patch("The color is blue.", "The color is red.")])
    assert not r.ok
    assert "ambiguous" in r.reason
    assert r.content == _DOC          # original untouched
    assert r.rejected_index == 0


def test_missing_match_rejected():
    r = D.apply_patches(_DOC, [D.Patch("NONEXISTENT TEXT", "x")])
    assert not r.ok
    assert "not found" in r.reason
    assert r.content == _DOC


def test_atomicity_later_failure_rolls_back_all():
    # First patch is valid, second fails → NOTHING is applied.
    r = D.apply_patches(_DOC, [
        D.Patch("Status: draft.", "Status: final."),
        D.Patch("NONEXISTENT", "x"),
    ])
    assert not r.ok
    assert r.rejected_index == 1
    assert r.content == _DOC          # the valid first patch is rolled back too
    assert r.applied == 0


def test_empty_old_str_rejected():
    r = D.apply_patches(_DOC, [D.Patch("", "x")])
    assert not r.ok
    assert "empty" in r.reason


def test_no_patches_rejected():
    r = D.apply_patches(_DOC, [])
    assert not r.ok
    assert r.content == _DOC


def test_noop_replace_is_not_changed():
    # old == new: applies but the content is identical → changed False.
    r = D.apply_patches(_DOC, [D.Patch("Status: draft.", "Status: draft.")])
    assert r.ok
    assert r.changed is False


def test_apply_never_raises():
    r = D.apply_patches(None, None)   # type: ignore[arg-type]
    assert not r.ok


# ---- diff_summary ---------------------------------------------------------
def test_diff_summary_counts_and_changed():
    d = D.diff_summary(_DOC, _DOC.replace("draft", "final"))
    assert d["added"] == 1 and d["removed"] == 1 and d["changed"] is True
    assert d["hunks"]


def test_diff_summary_no_change():
    d = D.diff_summary(_DOC, _DOC)
    assert d["changed"] is False
    assert d["added"] == 0 and d["removed"] == 0


# ---- build_patches (injected structured seam) ----------------------------
def _stub(obj):
    async def fn(schema, msgs, **kw):
        return _Res(obj)
    return fn


def test_build_patches_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: False)
    out = _run(D.build_patches("make it navy", _DOC,
                               structured_fn=_stub({"patches": [{"old_str": "a",
                                                                 "new_str": "b"}]})))
    assert out == []


def test_build_patches_parses_schema(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)
    out = _run(D.build_patches(
        "finalize the status", _DOC,
        structured_fn=_stub({"patches": [
            {"old_str": "Status: draft.", "new_str": "Status: final.",
             "replace_all": False}]})))
    assert len(out) == 1
    assert isinstance(out[0], D.Patch)
    # And it applies cleanly on the same document.
    r = D.apply_patches(_DOC, out)
    assert r.ok and "final" in r.content


def test_build_patches_fail_open_on_bad_obj(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)
    out = _run(D.build_patches("edit", _DOC, structured_fn=_stub("not a dict")))
    assert out == []


def test_build_patches_fail_open_on_raise(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)

    async def boom(schema, msgs, **kw):
        raise RuntimeError("llm down")
    out = _run(D.build_patches("edit", _DOC, structured_fn=boom))
    assert out == []


def test_end_to_end_nl_edit_to_new_content(monkeypatch):
    # NL instruction → patches (stubbed) → atomic apply → new versioned content.
    monkeypatch.setattr(D, "enabled", lambda: True)
    patches = _run(D.build_patches(
        "change the heading", _DOC,
        structured_fn=_stub({"patches": [
            {"old_str": "# Report", "new_str": "# Quarterly Report"}]})))
    r = D.apply_patches(_DOC, patches)
    assert r.ok and r.changed
    assert "# Quarterly Report" in r.content
