"""Stage 2 — Pattern classifier.

Maps a problem statement onto one of the spec's 24 algorithm families.
Two-tier dispatch:

  1. **Cheap keyword + signature heuristics** — catches the obvious
     cases (palindrome, two-sum, BFS keywords, etc.) without an LLM
     round-trip. Hits ~60% of problems with high confidence.
  2. **LLM fallback** — when no rule fires confidently, ask the
     classifier_model. Single round-trip, ~200 ms on Groq, ~1s on
     OpenRouter free.

Output: [PatternMatch] with `confidence` in [0, 1].
"""
from __future__ import annotations

import logging
import re

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from .types import PatternMatch, ProblemSpec
from app.core.prompt import fill

log = logging.getLogger(__name__)

# All 24 families from Architecture2.md §4. Order matters for the
# heuristic: more specific patterns are matched first so "sliding
# window" wins over plain "array".
_FAMILIES: list[str] = [
    "arrays/two-pointer",
    "sliding-window",
    "prefix-sum",
    "binary-search",
    "hashing",
    "monotonic-stack",
    "monotonic-queue",
    "heap/top-k",
    "linked-list",
    "trees",
    "tries",
    "graphs",
    "dp",
    "greedy",
    "backtracking",
    "divide-and-conquer",
    "bit-manipulation",
    "math/number-theory",
    "strings",
    "segment-tree-bit",
    "game-theory",
    "randomized",
    "union-find",
    "general",
]

# Keyword → family heuristics. Each entry is `(family, regex)`. The
# first match wins, so more-specific cues are listed before general
# ones — e.g. an explicit "two pointers" mention should beat a generic
# "sorted array ... find" that would otherwise grab binary-search.
_HEURISTICS: list[tuple[str, re.Pattern]] = [
    ("arrays/two-pointer",  re.compile(r"\b(two\s+pointers?|three\s+pointers?|opposite\s+ends?)\b", re.I)),
    ("sliding-window",      re.compile(r"\b(sliding\s*window|longest\s+substring|max\s+(?:sum|average)\s+of\s+subarray)\b", re.I)),
    ("prefix-sum",          re.compile(r"\b(prefix\s*sum|subarray\s+sum\s+equals?|range\s+sum)\b", re.I)),
    ("binary-search",       re.compile(r"\b(binary\s*search|find\s+(?:peak|rotation|target)|sorted\s+array.+(?:find|search))\b", re.I)),
    ("trees",               re.compile(r"\b(binary\s*tree|BST|root\s+of|tree\s+node|preorder|inorder|postorder|LCA|level\s+order)\b", re.I)),
    ("tries",               re.compile(r"\b(trie|prefix\s+tree|implement\s+trie|autocomplete)\b", re.I)),
    ("graphs",              re.compile(r"\b(graph|dijkstra|bellman\s*-?\s*ford|floyd|topological\s+sort|MST|kruskal|prim|SCC)\b", re.I)),
    ("union-find",          re.compile(r"\b(union\s*find|disjoint\s+set|connected\s+components|number\s+of\s+islands)\b", re.I)),
    ("monotonic-stack",     re.compile(r"\b(monotonic\s+stack|next\s+greater|daily\s+temperatures|largest\s+rectangle)\b", re.I)),
    ("monotonic-queue",     re.compile(r"\b(monotonic\s+queue|sliding\s+window\s+maximum)\b", re.I)),
    ("heap/top-k",          re.compile(r"\b(heap|priority\s+queue|top[\-\s]?k|kth\s+largest|kth\s+smallest|merge\s+k\s+sorted)\b", re.I)),
    ("linked-list",         re.compile(r"\b(linked\s*list|ListNode|reverse\s+(?:a\s+)?linked|cycle\s+in\s+linked)\b", re.I)),
    ("dp",                  re.compile(r"\b(dynamic\s+programming|dp\s*\[|knapsack|edit\s+distance|longest\s+common\s+subsequence|LCS|coin\s+change|climbing\s+stairs)\b", re.I)),
    ("backtracking",        re.compile(r"\b(backtrack|permutations|combinations|generate\s+all|n[\-\s]?queens|sudoku\s+solver|word\s+break)\b", re.I)),
    ("greedy",              re.compile(r"\b(greedy|jump\s+game|gas\s+station|interval\s+scheduling)\b", re.I)),
    ("bit-manipulation",    re.compile(r"\b(bit\s*manipulation|XOR|single\s+number|hamming|bitwise)\b", re.I)),
    ("strings",             re.compile(r"\b(palindrom|anagram|KMP|Z[\-\s]?algorithm|Rabin[\-\s]?Karp|suffix\s+array|Manacher)\b", re.I)),
    ("hashing",             re.compile(r"\b(two\s*sum|three\s*sum|hash\s*(?:map|set)|frequency|count\s+occurrence)\b", re.I)),
    ("math/number-theory",  re.compile(r"\b(gcd|lcm|prime|sieve|modular|number\s+theory|combinatori)\b", re.I)),
    ("segment-tree-bit",    re.compile(r"\b(segment\s+tree|fenwick|binary\s+indexed\s+tree|BIT|range\s+update)\b", re.I)),
    ("game-theory",         re.compile(r"\b(game\s+theory|minimax|nim|stone\s+game)\b", re.I)),
    ("randomized",          re.compile(r"\b(reservoir\s+sampling|randomized|random\s+pick)\b", re.I)),
    ("divide-and-conquer",  re.compile(r"\b(divide\s+and\s+conquer|merge\s+sort|quick\s*sort)\b", re.I)),
]

_CLASSIFIER_PROMPT = """You are a pattern classifier for algorithm interview problems.

Given the problem below, pick exactly one family from this list:
{families}

Reply in this exact format on a single line, no prose:
FAMILY: <family> | CONFIDENCE: <0.0-1.0> | RATIONALE: <one short sentence>

PROBLEM:
{statement}
"""

async def classify(problem: ProblemSpec) -> PatternMatch:
    """Pick the best family for `problem`. Heuristic first, LLM second."""
    if not problem.statement.strip():
        return PatternMatch(family="general", confidence=0.0, rationale="empty problem")

    # 1. Heuristics.
    for family, regex in _HEURISTICS:
        if regex.search(problem.statement):
            return PatternMatch(
                family=family,
                confidence=0.75,
                rationale=f"keyword match in problem ({regex.pattern[:40]}…)",
            )

    # 2. LLM fallback. Cheap classifier model.
    classifier_model = cfg.llm.classifier_model or cfg.llm.model
    prompt = fill(_CLASSIFIER_PROMPT, 
        families="\n  - ".join(_FAMILIES),
        statement=problem.statement[:3000],   # context cap
    )
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=classifier_model,
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.pattern},
        )
    except LLMError as exc:
        log.warning("pattern classifier LLM call failed: %s", exc)
        return PatternMatch(
            family="general", confidence=0.1, rationale="LLM unavailable, fell back to general"
        )

    return _parse_response(raw)

def _parse_response(raw: str) -> PatternMatch:
    """Parse `FAMILY: x | CONFIDENCE: y | RATIONALE: z` lenient-ly."""
    line = (raw or "").strip().splitlines()[0] if raw else ""
    family = "general"
    confidence = 0.4
    rationale = ""
    m = re.search(r"FAMILY\s*:\s*([a-zA-Z0-9/\-\s]+)", line, re.I)
    if m:
        proposed = m.group(1).strip().lower()
        # Map to canonical entry — exact match first, then substring.
        if proposed in _FAMILIES:
            family = proposed
        else:
            for canon in _FAMILIES:
                if canon in proposed or proposed in canon:
                    family = canon
                    break
    m = re.search(r"CONFIDENCE\s*:\s*([0-9.]+)", line, re.I)
    if m:
        try:
            confidence = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    m = re.search(r"RATIONALE\s*:\s*(.+)", line, re.I)
    if m:
        rationale = m.group(1).strip()
    return PatternMatch(family=family, confidence=confidence, rationale=rationale)
