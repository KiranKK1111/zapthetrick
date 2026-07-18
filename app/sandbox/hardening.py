"""Sandbox hardening (roadmap Phase 4 #19): pre-execution escape/egress detection.

The sandbox already caps CPU/memory/time/output and, on the `namespace` (bwrap)
backend, blocks network + gives a read-only OS. But the `rlimit` and
`subprocess` backends are HONEST about NOT isolating the network or the
filesystem mounts — so on those, a generated script that opens a socket or reads
`/etc/shadow` would actually reach out. This module statically scans a script for
those escape/egress attempts BEFORE it runs and, on a non-isolated backend,
blocks the high-severity ones. Deterministic, language-aware (py/js/bash), and
fail-open — a scan error never blocks a legitimate run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# severity: 3 = block-worthy on a non-isolated backend, 2 = notable, 1 = info.
_PATTERNS: list[tuple[str, int, str, re.Pattern]] = [
    # ── network egress ────────────────────────────────────────────────────
    ("egress", 3, "raw socket", re.compile(r"\bsocket\.socket\s*\(")),
    ("egress", 3, "http client", re.compile(
        r"\b(?:urllib\.request|urllib2|requests\.|http\.client|httpx\.|aiohttp\.|"
        r"fetch\s*\(|XMLHttpRequest|net\.connect|https?\.request)")),
    ("egress", 3, "shell net tool", re.compile(
        r"""(?:^|[;&|`$('"\s])(?:curl|wget|nc|ncat|netcat|telnet|ssh|scp|ftp|"""
        r"rsync)\b")),
    ("egress", 2, "mail/ftp lib", re.compile(r"\b(?:smtplib|ftplib|poplib|imaplib)\b")),
    ("egress", 2, "dns lookup", re.compile(r"\b(?:socket\.gethostby|dns\.resolver)")),
    # ── filesystem / secret exfiltration ─────────────────────────────────
    ("secret", 3, "reads system secrets", re.compile(
        r"/etc/(?:shadow|passwd|sudoers)|/root/|~?/\.ssh/|"
        r"\.aws/credentials|\.env\b|id_rsa")),
    ("path-escape", 2, "path traversal", re.compile(r"\.\./\.\./|\.\.\\\.\.\\")),
    ("secret", 2, "dumps environment", re.compile(
        r"os\.environ(?:\.copy\(\)|\b(?!\s*\.get))|process\.env\b|printenv\b")),
    # ── privilege / breakout ─────────────────────────────────────────────
    ("breakout", 3, "privilege escalation", re.compile(
        r"(?:^|[;&|`$(\s])sudo\b|\bsetuid\b|\bchroot\b|\bmount\s")),
    ("breakout", 2, "native-code loader", re.compile(r"\bctypes\.|cdll\b|LoadLibrary")),
    ("breakout", 2, "proc/kernel poke", re.compile(r"/proc/(?:self|sys)|/dev/(?:mem|kmem)")),
    # ── resource abuse ───────────────────────────────────────────────────
    ("fork-bomb", 3, "fork bomb", re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
]

# Comment prefixes stripped before scanning, so a warning IN a comment doesn't
# trip the scanner (per language).
_COMMENT = re.compile(r"(?m)(^\s*#.*$)|(^\s*//.*$)")


@dataclass
class Finding:
    category: str
    severity: int
    detail: str
    match: str


@dataclass
class HardeningReport:
    findings: list[Finding]
    max_severity: int
    blocked: bool
    net_isolated: bool

    def as_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "max_severity": self.max_severity,
            "net_isolated": self.net_isolated,
            "findings": [
                {"category": f.category, "severity": f.severity,
                 "detail": f.detail, "match": f.match}
                for f in self.findings
            ],
        }


def scan(code: str) -> list[Finding]:
    """Static scan for escape/egress attempts. Deterministic; comments ignored."""
    out: list[Finding] = []
    if not code:
        return out
    try:
        body = _COMMENT.sub("", code)
        for category, sev, detail, pat in _PATTERNS:
            m = pat.search(body)
            if m:
                out.append(Finding(category=category, severity=sev, detail=detail,
                                   match=m.group(0).strip()[:60]))
    except Exception:  # noqa: BLE001 — never let a scan error block a run
        return out
    return out


def assess(code: str, *, net_isolated: bool) -> HardeningReport:
    """Assess a script. On a NON-isolated backend a severity-3 finding blocks the
    run (the OS wouldn't contain it); on the namespace backend nothing is blocked
    (bwrap already denies network + read-only OS), findings are advisory only."""
    findings = scan(code)
    max_sev = max((f.severity for f in findings), default=0)
    blocked = (not net_isolated) and max_sev >= 3
    return HardeningReport(findings=findings, max_severity=max_sev,
                           blocked=blocked, net_isolated=net_isolated)


__all__ = ["Finding", "HardeningReport", "scan", "assess"]
