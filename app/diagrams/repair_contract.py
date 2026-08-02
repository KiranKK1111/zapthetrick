"""The Mermaid syntax-repair contract — prompt + reply unwrapping.

Domain knowledge about mermaid, not HTTP plumbing, so it lives with the
other diagram logic. It used to sit in `app/api/routes_mermaid.py`, which
forced `app/diagrams` to import a ROUTE module to reuse it — a domain
package depending on the transport layer, backwards from every other
package here. Both the route and the verifier now import it from here.
"""
from __future__ import annotations

import re


_REPAIR_PROMPT = (
    "The following Mermaid diagram failed compilation.\n\n"
    "Parser error:\n{error}\n\n"
    "Mermaid source:\n```mermaid\n{source}\n```\n\n"
    # NOTE: this string goes through `str.format`, so every literal brace below
    # must be doubled — a stray `{` raises KeyError and silently costs the repair.
    "Fix ONLY the syntax. Do NOT change the architecture, the nodes, the "
    "labels' meaning, or the layout direction. Common REAL fixes: wrap a label "
    "containing `(`, `)`, `\"`, `|`, `{{` or `}}` in double quotes "
    "(A[\"Fetch (REST)\"]); close every `subgraph` with `end`; give every link "
    "an arrowhead or terminator (`A --> B` or `A --- B`, never `A -- B`); use at "
    "least two dashes (`-->`, not `->`); declare nodes as `ID[\"Label\"]`.\n"
    "Do NOT 'fix' these — they are VALID mermaid: `--->` and `----` (extra "
    "dashes just set the rank distance), and `:`, `#`, `<`, `>`, `;`, `&`, `%` "
    "inside an unquoted label.\n"
    "Return ONLY the corrected Mermaid code, with no fences and no commentary."
)


def _strip_fences(text: str) -> str:
    """The model sometimes fences the reply anyway — unwrap it."""
    t = (text or "").strip()
    m = re.search(r"```(?:mermaid)?\s*\n(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    return t


__all__ = ["_REPAIR_PROMPT", "_strip_fences"]
