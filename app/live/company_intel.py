"""Company intelligence — the OrgBrief (vNext §4.12, Stage 6 Component I).

A company URL at session setup becomes a typed **OrgBrief** the interview can
answer org-questions from ("what do you know about us?", "why here?") in the
candidate's voice: identity, product/service, offerings, stack, focus, trajectory
(stated / recent / outlook), with a freshness stamp per fact. The brief is cached
per company (7-day freshness) and rides the session's L3 context.

This module owns the SECURITY-critical + deterministic pieces:
  * **`is_safe_url(url)`** — an SSRF guard that MUST run before any crawl:
    http(s) only, and never localhost / private / loopback / link-local / the
    cloud-metadata endpoint (169.254.169.254). Literal-IP and hostname checks are
    deterministic; DNS resolution is best-effort on top.
  * the typed **`OrgBrief`** + its band-shaped `directive()`;
  * **org-question detection** (`is_org_question`) — SEMANTIC via the
    `org_question` gate (authority), a cue list only as cold-start fallback;
  * a per-company **cache** with a 7-day TTL.

The actual crawl + web research is INJECTED (`crawl_fn`) — network-bound, it runs
on the pod; this module orchestrates + guards it, so the logic is unit-tested
with no network. Fail-open throughout. Flag-gated (`live.company_intel`, OFF).
"""
from __future__ import annotations

import ipaddress
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

log = logging.getLogger(__name__)

_LOCK = threading.RLock()


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "company_intel", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# SSRF guard (security-critical)
# --------------------------------------------------------------------------- #
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}
_LOCAL_SUFFIXES = (".local", ".internal", ".localhost")


def _ip_is_public(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified)
    except ValueError:
        return False


def is_safe_url(url: str) -> bool:
    """SSRF guard — True only for a public http(s) URL. Blocks non-http(s)
    schemes, localhost / *.local / *.internal, literal private/loopback/link-local
    IPs, and the cloud-metadata endpoint. A hostname is resolved best-effort;
    if it maps to any non-public address it is rejected. Never raises → False on
    anything it can't prove safe (deny by default)."""
    try:
        u = urlparse((url or "").strip())
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        host = u.hostname.lower().rstrip(".")
        if host in _BLOCKED_HOSTNAMES:
            return False
        if any(host == s.lstrip(".") or host.endswith(s)
               for s in _LOCAL_SUFFIXES):
            return False
        # Literal IP → check directly.
        try:
            ipaddress.ip_address(host)
            return _ip_is_public(host)
        except ValueError:
            pass
        # Hostname → best-effort resolve; reject if ANY resolved addr is non-public.
        try:
            import socket
            infos = socket.getaddrinfo(host, u.port or 443,
                                       proto=socket.IPPROTO_TCP)
            addrs = {i[4][0] for i in infos}
            if addrs and all(_ip_is_public(a) for a in addrs):
                return True
            return False if addrs else True   # unresolved → allow (crawl will fail safe)
        except Exception:  # noqa: BLE001 — DNS error → don't hard-block a real host
            return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# OrgBrief
# --------------------------------------------------------------------------- #
@dataclass
class OrgBrief:
    company: str
    identity: str = ""                      # one-line "who they are"
    product: str = ""                       # primary product/service
    offerings: list[str] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    trajectory: dict = field(default_factory=dict)  # {stated, recent, outlook}
    freshness: dict = field(default_factory=dict)   # fact -> iso timestamp
    source_url: str = ""
    fetched_at: float = 0.0

    def as_dict(self) -> dict:
        return {"company": self.company, "identity": self.identity,
                "product": self.product, "offerings": list(self.offerings),
                "stack": list(self.stack), "focus": list(self.focus),
                "trajectory": dict(self.trajectory),
                "freshness": dict(self.freshness), "source_url": self.source_url}

    def directive(self) -> str:
        """A compact, band-shaped L3 context line for answering an org-question in
        the candidate's own voice (facts only — the model phrases them)."""
        try:
            bits = []
            if self.identity:
                bits.append(self.identity)
            if self.product:
                bits.append(f"Product: {self.product}")
            if self.offerings:
                bits.append("Offerings: " + ", ".join(self.offerings[:5]))
            if self.stack:
                bits.append("Stack: " + ", ".join(self.stack[:8]))
            if self.focus:
                bits.append("Focus: " + ", ".join(self.focus[:5]))
            traj = self.trajectory or {}
            tline = " · ".join(f"{k}: {v}" for k, v in traj.items() if v)
            if tline:
                bits.append("Trajectory — " + tline)
            if not bits:
                return ""
            return (f"About {self.company} (use these FACTS to answer, in your "
                    "own words; do not invent beyond them): " + " | ".join(bits))
        except Exception:  # noqa: BLE001
            return ""


# --------------------------------------------------------------------------- #
# Org-question detection (semantic-first)
# --------------------------------------------------------------------------- #
_ORG_CUES = re.compile(
    r"\b(our company|our product|our platform|our business|our mission|our "
    r"customers|our (?:tech )?stack|about us|work here|why here|what do we|"
    r"what does .* do|who are (?:your|our) customers)\b", re.I)


def is_org_question(text: str) -> bool:
    """Whether the interviewer's question is ABOUT the company (→ answer from the
    OrgBrief). SEMANTIC-first: the `org_question` gate is the authority; the cue
    regex is only the cold-start fallback. Never raises."""
    try:
        t = (text or "").strip()
        if not t:
            return False
        try:
            from app.semantics import gates
            verdict = gates.matches("org_question", t)
            if verdict is not None:
                return bool(verdict)
        except Exception:  # noqa: BLE001
            pass
        return bool(_ORG_CUES.search(t))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Per-company cache (7-day freshness)
# --------------------------------------------------------------------------- #
_cache: dict[str, tuple[OrgBrief, float]] = {}


def _ttl_s() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "org_brief_ttl_days", 7.0)) * 86_400.0
    except Exception:  # noqa: BLE001
        return 7.0 * 86_400.0


def _key(company: str) -> str:
    return re.sub(r"\s+", " ", (company or "").strip().lower())


def cache_get(company: str, *, now: float | None = None) -> OrgBrief | None:
    now = time.time() if now is None else now
    with _LOCK:
        row = _cache.get(_key(company))
    if row is None:
        return None
    brief, stamped = row
    if now - stamped >= _ttl_s():
        return None                          # stale → re-fetch
    return brief


def cache_put(company: str, brief: OrgBrief, *, now: float | None = None) -> None:
    now = time.time() if now is None else now
    with _LOCK:
        _cache[_key(company)] = (brief, now)


def reset_for_tests() -> None:
    with _LOCK:
        _cache.clear()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
CrawlFn = Callable[[str], "dict | Awaitable[dict]"]


async def build_brief(company: str, *, url: str | None = None,
                      crawl_fn: CrawlFn | None = None,
                      now: float | None = None) -> OrgBrief | None:
    """Build (or serve from cache) the OrgBrief for `company`. A fresh cache hit
    returns instantly. Else, if a `url` + `crawl_fn` are given AND the url passes
    the SSRF guard, crawl → assemble → cache. Returns None when disabled / no
    data / any error (fail-open — the session falls back to the deterministic
    research brief)."""
    if not enabled():
        return None
    try:
        cached = cache_get(company, now=now)
        if cached is not None:
            return cached
        if not (url and crawl_fn):
            return None
        if not is_safe_url(url):
            log.info("company_intel: refused unsafe URL %r (SSRF guard)", url)
            return None
        raw = crawl_fn(url)
        import inspect
        if inspect.isawaitable(raw):
            raw = await raw
        brief = _assemble(company, url, raw or {}, now=now)
        cache_put(company, brief, now=now)
        return brief
    except Exception as exc:  # noqa: BLE001
        log.info("company_intel.build_brief failed: %s", exc)
        return None


def _assemble(company: str, url: str, raw: dict, *,
              now: float | None = None) -> OrgBrief:
    """Map a crawl/research result dict into a typed OrgBrief. Missing fields are
    simply omitted; per-fact freshness defaults to the fetch time."""
    ts = time.time() if now is None else now

    def _list(v):
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else (
            [str(v)] if v else [])

    return OrgBrief(
        company=company,
        identity=str(raw.get("identity") or ""),
        product=str(raw.get("product") or ""),
        offerings=_list(raw.get("offerings")),
        stack=_list(raw.get("stack")),
        focus=_list(raw.get("focus")),
        trajectory=dict(raw.get("trajectory") or {}),
        freshness=dict(raw.get("freshness") or {}),
        source_url=url, fetched_at=ts)


__all__ = ["enabled", "is_safe_url", "OrgBrief", "is_org_question",
           "cache_get", "cache_put", "build_brief", "reset_for_tests"]
