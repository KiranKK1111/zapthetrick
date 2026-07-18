"""In-loop web tools for the agent — WebSearch / WebFetch parity (P2-6).

Claude can look things up mid-task (an API, a library version, an error
message). This exposes that to our agent loop as two tools:

  • web_search(query)  — DuckDuckGo results (reuses app.tools.web_search), no key.
  • web_fetch(url)     — fetch a page and return readable text.

Two safety properties baked in (the untrusted-content side of P2-12):
  1. SSRF guard — only http/https, and the resolved host must be PUBLIC (no
     loopback / private / link-local / metadata IPs). The agent runs on the VPS,
     so it must not be trickable into hitting internal services.
  2. Untrusted-content guard — fetched/searched text is wrapped in an explicit
     boundary telling the model to treat it as DATA, not instructions (defends
     against prompt injection embedded in a page).

Both tools are best-effort and return a STRING (fed straight back to the model).
Network/parse failures come back as a clear `ERROR:`/note, never an exception.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

_MAX_FETCH_BYTES = 600_000
_MAX_TEXT_CHARS = 12_000
_FETCH_TIMEOUT = 20.0


def _banner() -> str:
    from app.agent.safety import UNTRUSTED_BANNER
    return UNTRUSTED_BANNER


# ── SSRF guard ──────────────────────────────────────────────────────────────
def _is_public_host(host: str) -> bool:
    """True only if every resolved address for `host` is a public IP."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def check_url(url: str) -> tuple[bool, str]:
    """(allowed, reason). Enforces scheme + public-host (anti-SSRF)."""
    try:
        p = urlparse((url or "").strip())
    except Exception:  # noqa: BLE001
        return False, "could not parse URL"
    if p.scheme not in ("http", "https"):
        return False, "only http/https URLs are allowed"
    host = p.hostname or ""
    if not host:
        return False, "URL has no host"
    # Block obvious internal names + cloud metadata endpoints up front.
    low = host.lower()
    if low in ("localhost", "metadata", "metadata.google.internal") \
            or low.endswith(".local") or low.endswith(".internal"):
        return False, "refusing to fetch an internal/metadata host"
    if not _is_public_host(host):
        return False, "host resolves to a non-public address (blocked)"
    return True, ""


# ── HTML → text ───────────────────────────────────────────────────────────────
_SCRIPT_STYLE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>",
                           re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(html: str) -> str:
    """A dependency-free readable-text extraction (strip scripts/tags/entities)."""
    import html as _html

    s = _SCRIPT_STYLE.sub(" ", html or "")
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|li|h[1-6]|tr|section|article)>", "\n", s, flags=re.I)
    s = _TAG.sub("", s)
    s = _html.unescape(s)
    s = _WS.sub(" ", s)
    s = _BLANKS.sub("\n\n", s)
    return s.strip()


# ── tools ─────────────────────────────────────────────────────────────────────
async def web_search(query: str, *, max_results: int = 5) -> str:
    """Search the public web (DuckDuckGo). Returns a compact result list."""
    q = (query or "").strip()
    if not q:
        return "ERROR: empty search query."
    try:
        from app.tools.web_search import search
        hits = await search(query=q, max_results=max(1, min(max_results, 10)))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: web search failed: {exc}"
    if not hits:
        return f"(no results for {q!r})"
    lines = [_banner(), f"Search results for {q!r}:"]
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        url = (h.get("url") or "").strip()
        snippet = (h.get("snippet") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title}\n   {url}\n   {snippet[:300]}")
    return "\n".join(lines)


async def web_fetch(url: str, *, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Fetch a URL and return its readable text (SSRF-guarded, size-capped)."""
    url = (url or "").strip()
    ok, reason = check_url(url)
    if not ok:
        return f"ERROR: {reason}: {url}"
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return "ERROR: httpx is not available to fetch URLs."
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": "ZapTheTrick-Agent/1.0"},
        ) as client:
            resp = await client.get(url)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: could not fetch {url}: {exc}"
    # Re-check the FINAL URL after redirects (a public URL can 30x to internal).
    final_ok, final_reason = check_url(str(resp.url))
    if not final_ok:
        return f"ERROR: redirected to a blocked host ({final_reason})."
    if resp.status_code >= 400:
        return f"ERROR: {url} returned HTTP {resp.status_code}."
    ctype = resp.headers.get("content-type", "")
    raw = resp.content[:_MAX_FETCH_BYTES]
    try:
        text = raw.decode(resp.encoding or "utf-8", errors="replace")
    except (LookupError, TypeError):
        text = raw.decode("utf-8", errors="replace")
    body = html_to_text(text) if ("html" in ctype or "<html" in text[:500].lower()) \
        else text
    if not body.strip():
        return f"(fetched {url} but found no readable text)"
    clipped = body[:max(500, max_chars)]
    more = "\n…(truncated)" if len(body) > len(clipped) else ""
    return (f"{_banner()}Content of {resp.url}:\n\n{clipped}{more}")


__all__ = ["web_search", "web_fetch", "check_url", "html_to_text"]
