"""
Conservative transcript repair (live-conversational-intelligence R11).

Token level: fixes a likely STT mis-recognition of a DOMAIN term using the
session vocabulary + topic-graph context — but only when a token is clearly
not a real word AND a phonetically near-identical domain term is the obvious
fix.

Phrase level: real Whisper/ASR errors often melt a domain term into a run of
common words ("my service" → "microservices", "cube net is ingress" →
"kubernetes ingress"). A sliding 2-3 word n-gram window is fuzzy-matched
against single- AND multi-word domain terms using a normalized similarity on
the joined lowercase string plus a metaphone-like phonetic gate (both stdlib /
inline — no new dependencies). A window is replaced only when similarity is
high, the window is not part of an exact vocabulary match, and the word count
of the replacement differs from the window by at most one.

It NEVER introduces a proper noun / model / acronym that isn't phonetically
present, and NEVER swaps a real English/technical word (mirrors the
`agent.predict` cleanup contract). Deterministic + fail-open: on any error
returns the transcript unchanged.
"""
from __future__ import annotations

import difflib
import re

# A small built-in domain vocabulary; callers extend it with the STT seed +
# the live topic graph so repair adapts to the actual interview.
_BUILTIN_VOCAB = {
    "kafka", "kubernetes", "docker", "redis", "postgres", "postgresql",
    "mongodb", "cassandra", "rabbitmq", "nginx", "graphql", "grpc", "spring",
    "hibernate", "react", "angular", "flutter", "python", "java", "javascript",
    "typescript", "golang", "terraform", "ansible", "prometheus", "grafana",
    "elasticsearch", "microservices", "partitions", "offsets", "consumer",
    "polymorphism", "concurrency", "mutex", "semaphore", "throughput",
}

# Built-in multi-word domain phrases (common interview tech phrases). These
# are fuzzy targets for the phrase-level pass and exact-match protected when
# already present in the transcript.
_BUILTIN_PHRASES = {
    "kubernetes ingress", "consumer groups", "event ordering",
    "load balancer", "message queue", "dependency injection",
    "garbage collection", "connection pool", "circuit breaker",
    "rate limiting", "primary key", "foreign key", "unit testing",
    "design patterns", "system design", "machine learning",
    "neural network", "distributed systems", "eventual consistency",
    "horizontal scaling",
}

# Common English words that must never be "corrected" away (token level). At
# phrase level an n-gram made ONLY of these may still be replaced, but only
# when the whole n-gram jointly matches a domain term strongly (stricter
# threshold) — the "my service" → "microservices" case.
_COMMON = {
    "the", "and", "for", "with", "what", "how", "why", "when", "where", "who",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "this", "that", "those", "these", "your", "you", "about", "into",
    "from", "have", "has", "will", "cohesion", "coupling", "binding", "innovation",
    "design", "system", "data", "code", "test", "build", "scale", "service",
    "my", "our", "its", "not", "all", "any", "level", "unit",
}

# Phrase-pass tuning (normalized distances in [0, 1]).
_NGRAM_MAX_DIST = 0.34      # joined-string distance for a normal n-gram
_NGRAM_COMMON_DIST = 0.30   # stricter bar when every window word is common
_PHONETIC_MAX_DIST = 0.35   # metaphone-like gate on the joined strings
_MIN_JOINED_LEN = 5         # ignore tiny windows ("is my", ...)
_MIN_TERM_LEN = 6           # ignore tiny targets
_MIN_SINGLE_TERM_LEN = 8    # single-word terms eligible as phrase targets

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _best_vocab_match(token: str, vocab: set[str]) -> str | None:
    """Nearest vocab term within a conservative edit-distance budget, or None."""
    t = token.lower()
    if len(t) < 4 or t in vocab or t in _COMMON:
        return None
    # Budget scales gently with length; never more than 2 edits.
    budget = 1 if len(t) <= 6 else 2
    best, best_d = None, budget + 1
    for term in vocab:
        if abs(len(term) - len(t)) > budget or len(term) < 4:
            continue
        d = _levenshtein(t, term)
        if d < best_d and d <= budget:
            best, best_d = term, d
    # Require a real near-miss (not a 0-distance already-correct token).
    return best if best and best != t else None


# ---- phrase-level machinery ---------------------------------------------
def _norm_dist(a: str, b: str) -> float:
    """Normalized similarity distance on strings (0 identical … 1 disjoint)."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def _lev_dist(a: str, b: str) -> float:
    """Levenshtein distance normalized by the longer string's length."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    return _levenshtein(a, b) / max(len(a), len(b))


_VOWELISH = frozenset("aeiouyhw")


def _phonetic(s: str) -> str:
    """Tiny metaphone-like consonant skeleton (inline, deterministic)."""
    t = re.sub(r"[^a-z]", "", s.lower())
    for src, dst in (("ph", "f"), ("gh", "g"), ("ck", "k"), ("qu", "kw")):
        t = t.replace(src, dst)
    out: list[str] = []
    for i, ch in enumerate(t):
        if ch == "c":
            ch = "s" if t[i + 1:i + 2] in ("e", "i", "y") else "k"
        elif ch == "q":
            ch = "k"
        elif ch in ("z", "x"):
            ch = "s"
        if i and ch in _VOWELISH:
            continue  # drop non-leading vowels / soft consonants
        if out and out[-1] == ch:
            continue  # collapse repeats
        out.append(ch)
    return "".join(out)


def _commonish(w: str) -> bool:
    return w in _COMMON or (w.endswith("s") and w[:-1] in _COMMON)


def _same_inflection(ws: list[str], ts: list[str]) -> bool:
    """True when window and term are the same phrase modulo word endings
    ("unit tests" vs "unit testing") — grammar, not a mishearing."""
    if len(ws) != len(ts):
        return False
    for a, b in zip(ws, ts):
        if a == b:
            continue
        p = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            p += 1
        if p >= 3 and p >= min(len(a), len(b)) - 2:
            continue
        return False
    return True


def _phrase_pass(text: str, terms: set[str], phrases: set[str]) -> str:
    """Sliding 2-3 word n-gram fuzzy repair against domain terms/phrases."""
    tokens = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if len(tokens) < 2:
        return text
    words = [re.sub(r"[^a-z]", "", tok[0].lower()) for tok in tokens]

    targets = set(phrases)
    targets.update(t for t in terms
                   if " " not in t and len(t) >= _MIN_SINGLE_TERM_LEN)

    # Exact vocabulary matches are protected: never rewrite (part of) a span
    # that already spells a known term — keeps repair idempotent and keeps
    # "service level agreement"-style phrases intact.
    protected: set[int] = set()
    for term in phrases:
        tws = term.split()
        for i in range(len(words) - len(tws) + 1):
            if words[i:i + len(tws)] == tws:
                protected.update(range(i, i + len(tws)))
    for i, w in enumerate(words):
        if len(w) >= 4 and w in terms:
            protected.add(i)

    candidates: list[tuple[float, int, int, str]] = []
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            if any(j in protected for j in range(i, i + n)):
                continue
            # Never span punctuation — only whitespace between window tokens.
            if any(text[tokens[j][2]:tokens[j + 1][1]].strip()
                   for j in range(i, i + n - 1)):
                continue
            ws = words[i:i + n]
            jw = "".join(ws)
            if len(jw) < _MIN_JOINED_LEN:
                continue
            all_common = all(_commonish(w) for w in ws)
            threshold = _NGRAM_COMMON_DIST if all_common else _NGRAM_MAX_DIST
            for term in targets:
                tws = term.split()
                if abs(n - len(tws)) > 1:
                    continue  # replacement word count must differ by ≤ 1
                jt = term.replace(" ", "")
                if len(jt) < _MIN_TERM_LEN or jw == jt:
                    continue
                if set(tws) <= set(ws):
                    continue  # window already contains the full term
                if jt.startswith(jw) or jw.startswith(jt):
                    continue  # pure elongation ("unit test" → "unit testing")
                if _same_inflection(ws, tws):
                    continue  # grammatical variant, not a mishearing
                # A correct domain word inside the window may only take part
                # if it survives verbatim in the replacement term.
                if any(len(w) >= 4 and w in terms and not _commonish(w)
                       and w not in tws for w in ws):
                    continue
                # Equal word counts → every aligned word must be plausibly
                # similar (blocks "the design" → "system design").
                if len(ws) == len(tws) and any(
                        _norm_dist(a, b) >= 0.5 for a, b in zip(ws, tws)):
                    continue
                d = _norm_dist(jw, jt)
                if d >= threshold:
                    continue
                if _lev_dist(_phonetic(jw), _phonetic(jt)) >= _PHONETIC_MAX_DIST:
                    continue
                candidates.append((d, i, n, term))

    if not candidates:
        return text

    # Best (lowest-distance) candidates win; overlaps are dropped.
    candidates.sort(key=lambda c: (c[0], c[1], -c[2]))
    used: set[int] = set(protected)
    chosen: list[tuple[int, int, str]] = []
    for d, i, n, term in candidates:
        span = set(range(i, i + n))
        if span & used:
            continue
        used |= span
        chosen.append((i, n, term))

    # Apply right-to-left so char offsets stay valid.
    out = text
    for i, n, term in sorted(chosen, key=lambda c: -c[0]):
        start, end = tokens[i][1], tokens[i + n - 1][2]
        rep = term
        # Preserve leading capitalization of the original window.
        if tokens[i][0][:1].isupper():
            rep = rep[0].upper() + rep[1:]
        out = out[:start] + rep + out[end:]
    return out


# Standalone hesitation fillers the ASR faithfully transcribes ("can you...
# ahh... tell me"). Word-boundary only, so real words are never touched;
# "uh-huh"/"mm-hmm" are matched whole so their halves don't survive.
_FILLER_RE = re.compile(
    r"\b(?:uh-huh|mm-hmm|uh+|um+|ah+|er+m?|hm+|mm+|mhm)\b[,.!?]?\s*",
    re.IGNORECASE,
)


def strip_fillers(text: str) -> str:
    """Remove standalone hesitation fillers ('uh', 'um', 'ahh', 'hmm') from a
    transcript. They add noise for the LLM and mislead the completeness /
    continuation heuristics ("can you ahh" reads neutral instead of
    incomplete). Fail-open: returns the input on any error or if stripping
    would empty the text."""
    try:
        t = text or ""
        out = _FILLER_RE.sub("", t)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s+([?,.!;:])", r"\1", out).strip()
        return out if out else t
    except Exception:  # noqa: BLE001
        return text or ""


def repair(transcript: str, vocab=None, topic_graph=None) -> str:
    """Conservatively repair domain-term STT errors. Never raises."""
    text = transcript or ""
    if not text.strip():
        return text
    try:
        terms = set(_BUILTIN_VOCAB)
        phrases = set(_BUILTIN_PHRASES)
        if vocab:
            for v in vocab:
                w = str(v).strip().lower()
                if not w:
                    continue
                if " " in w:
                    w = re.sub(r"\s+", " ", w)
                    phrases.add(w)
                    for part in w.split():
                        if len(part) >= 4:
                            terms.add(part)
                else:
                    terms.add(w)
        if topic_graph is not None:
            try:
                for tp in topic_graph.topics():
                    t = re.sub(r"\s+", " ", str(tp).strip().lower())
                    tp_words = t.split()
                    if 2 <= len(tp_words) <= 3:
                        phrases.add(t)
                    for w in tp_words:
                        if len(w) >= 4:
                            terms.add(w)
            except Exception:  # noqa: BLE001
                pass

        def _fix(m: re.Match) -> str:
            token = m.group(0)
            match = _best_vocab_match(token, terms)
            if match is None:
                return token
            # Preserve leading capitalization of the original token.
            return match.capitalize() if token[:1].isupper() else match

        text = re.sub(r"[A-Za-z][A-Za-z'-]*", _fix, text)
        return _phrase_pass(text, terms, phrases)
    except Exception:  # noqa: BLE001
        return transcript or ""
