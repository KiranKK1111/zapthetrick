"""Phase 1b — subprocess-isolated rendering.

The worker entrypoint is a pure function (tested in-process), and one test spawns
a REAL worker process to prove crash-isolated rendering produces valid bytes.
"""
from __future__ import annotations

import io
import zipfile

from app.documents import render_isolated as ri

_MD = ("# Title\n\nA paragraph.\n\n## Section\n\n- a\n- b\n\n"
       "| X | Y |\n|---|---|\n| 1 | 2 |\n")


def test_render_payload_is_a_pure_picklable_result():
    """render_payload returns the picklable dict the pool ships back — verified
    in-process (no subprocess needed)."""
    res = ri.render_payload(_MD, "docx", title="Doc")
    assert set(res) == {"data", "mime", "ext", "val_meta"}
    assert res["ext"] == "docx"
    assert res["data"][:2] == b"PK"          # docx is a zip container
    # bytes + str payload → picklable across a process boundary.
    import pickle
    pickle.loads(pickle.dumps(res))


def test_render_payload_matches_direct_render():
    from app.documents.generators import render_document
    direct, _, _ = render_document(_MD, "md", title="Doc")
    res = ri.render_payload(_MD, "md", title="Doc")
    assert res["data"] == direct


def test_render_isolated_runs_in_a_real_subprocess():
    """Actually spawn a worker process and render — proves the isolation path
    yields valid, openable bytes. Torn down afterwards so no orphan lingers."""
    try:
        res = ri.render_isolated(_MD, "docx", title="Doc", timeout=45)
    finally:
        ri.shutdown_pool()
    assert res["ext"] == "docx"
    # A real .docx: a zip whose namelist has the wordprocessing document part.
    with zipfile.ZipFile(io.BytesIO(res["data"])) as zf:
        assert "word/document.xml" in zf.namelist()


def test_shutdown_pool_is_idempotent():
    ri.shutdown_pool()
    ri.shutdown_pool()  # no error when already down
    assert ri._POOL is None
