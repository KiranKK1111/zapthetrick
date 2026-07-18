"""Tool planning (agent-orchestration R2, task 2.2).

Pins Property 2: selection/order from the registry, permission gating (never
plans an ungranted tool), and route-around on failure.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.orchestration.tool_plan import plan_tools, route_around


@dataclass
class _Tool:
    name: str
    description: str = ""
    server: str = "srv"


_TOOLS = [
    _Tool("web_search", "search the web for documentation"),
    _Tool("run_sql", "execute a SQL query against the database"),
    _Tool("send_email", "send an email message"),
]


def test_selects_relevant_tools():
    plan = plan_tools("search the web for the latest docs", None, _TOOLS)
    assert "web_search" in plan.names()
    assert "send_email" not in plan.names()


def test_permission_gating_skips_ungranted():
    granted = {"web_search"}
    plan = plan_tools("search the web and run a sql query", None, _TOOLS,
                      is_granted=lambda t: t.name in granted)
    assert "web_search" in plan.names()
    assert "run_sql" not in plan.names()       # not granted → skipped
    assert "run_sql" in plan.skipped_ungranted


def test_route_around_drops_failed_tool():
    plan = plan_tools("search the web and run sql", None, _TOOLS)
    before = set(plan.names())
    plan2 = route_around(plan, "web_search")
    assert "web_search" not in plan2.names()
    assert set(plan2.names()) == before - {"web_search"}


def test_no_match_empty_plan():
    plan = plan_tools("just chat with me", None, _TOOLS)
    assert plan.names() == []


def test_reliability_breaks_ties_between_equally_relevant_tools():
    """Two tools match the request identically; the one that actually works wins."""
    from app.tools import reliability as rel

    @dataclass
    class _T:
        name: str
        description: str
        server: str = "s"

    tools = [_T("alpha_fetch", "fetch a record"), _T("beta_fetch", "fetch a record")]
    rel.reset()
    for _ in range(6):
        rel.record("alpha_fetch", False)   # known-bad
        rel.record("beta_fetch", True)     # known-good
    try:
        plan = plan_tools("fetch a record", None, tools)
        assert plan.names()[0] == "beta_fetch"   # reliable tool ranked first
    finally:
        rel.reset()


def test_relevance_still_dominates_reliability():
    """A reliable but irrelevant tool must never displace the relevant one."""
    from app.tools import reliability as rel

    @dataclass
    class _T:
        name: str
        description: str
        server: str = "s"

    tools = [_T("run_sql", "execute a sql query"), _T("web_search", "search the web")]
    rel.reset()
    for _ in range(6):
        rel.record("run_sql", False)       # relevant but flaky
        rel.record("web_search", True)     # reliable but off-topic
    try:
        plan = plan_tools("execute a sql query", None, tools)
        assert plan.names()[0] == "run_sql"  # relevance wins over reliability
    finally:
        rel.reset()
