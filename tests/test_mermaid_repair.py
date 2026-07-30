"""Mermaid LLM repair endpoint — syntax-only fix loop (non-LLM parts + mocked LLM)."""
import asyncio

import pytest

from app.api import routes_mermaid as rm


class TestStripFences:
    def test_unwraps_mermaid_fence(self):
        assert rm._strip_fences("```mermaid\nflowchart LR\nA --> B\n```") == \
            "flowchart LR\nA --> B"

    def test_unwraps_bare_fence(self):
        assert rm._strip_fences("```\nflowchart TD\nA --> B\n```") == \
            "flowchart TD\nA --> B"

    def test_plain_text_passthrough(self):
        assert rm._strip_fences("flowchart LR\nA --> B") == "flowchart LR\nA --> B"


class TestRepairEndpoint:
    def test_empty_source_no_llm_call(self):
        res = asyncio.run(rm.repair(rm.MermaidRepairRequest(source="  ")))
        assert res.changed is False

    def test_llm_fix_returned(self, monkeypatch):
        async def fake_complete(messages, options=None):
            # The prompt must carry the parser error and the source.
            content = messages[0]["content"]
            assert "Unexpected token" in content
            assert "A[Fetch (REST)]" in content
            return '```mermaid\nflowchart LR\nA["Fetch (REST)"] --> B\n```'

        from app.core import llm_client
        monkeypatch.setattr(llm_client.llm, "complete", fake_complete)
        res = asyncio.run(rm.repair(rm.MermaidRepairRequest(
            source="flowchart LR\nA[Fetch (REST)] --> B",
            error="Unexpected token at line 2",
        )))
        assert res.changed is True
        assert 'A["Fetch (REST)"]' in res.source
        assert "```" not in res.source

    def test_llm_failure_is_best_effort(self, monkeypatch):
        async def boom(messages, options=None):
            raise RuntimeError("no route")

        from app.core import llm_client
        monkeypatch.setattr(llm_client.llm, "complete", boom)
        res = asyncio.run(rm.repair(rm.MermaidRepairRequest(
            source="flowchart LR\nA --> B", error="x")))
        assert res.changed is False
        assert res.source == "flowchart LR\nA --> B"

    def test_garbage_reply_rejected(self, monkeypatch):
        async def fake_complete(messages, options=None):
            return "ok"  # too short to be a diagram

        from app.core import llm_client
        monkeypatch.setattr(llm_client.llm, "complete", fake_complete)
        res = asyncio.run(rm.repair(rm.MermaidRepairRequest(
            source="flowchart LR\nA --> B", error="x")))
        assert res.changed is False


@pytest.mark.parametrize("path", ["/api/mermaid/repair"])
def test_route_registered(path):
    # Checked on the ROUTER, not via `app.main`. Importing `app.main` from a test
    # warms models and wires the live stack process-wide, and that state leaks
    # into whatever runs afterwards (it breaks `test_live_operability.py`'s
    # websocket tests). This file only got away with it because `test_mermaid_*`
    # happens to sort AFTER `test_live_*`.
    assert path in {r.path for r in rm.router.routes}


def test_router_is_mounted_in_main():
    from pathlib import Path
    main_py = Path(__file__).resolve().parents[1] / "app" / "main.py"
    source = main_py.read_text(encoding="utf-8", errors="replace")
    assert "routes_mermaid import router as mermaid_router" in source
    assert "include_router(mermaid_router)" in source
