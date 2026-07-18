"""Decide whether the user asked to GENERATE a downloadable document/file.

Used only to FLAG the assistant turn so the UI shows the document card / opens
the preview panel — not to change the answer. Understanding this intent is
delegated to the model (one small JSON call on the fast classifier model); there
are no keyword/regex rules. On any failure it falls back to "not a document
request" so a classifier hiccup never mis-flags an ordinary answer.
"""
from __future__ import annotations

import json
import logging
import re

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

log = logging.getLogger(__name__)

_FORMATS = {"pdf", "docx", "pptx", "xlsx", "csv", "json", "md", "txt", "zip"}

# --------------------------------------------------------------------------
# Deterministic fast-path for EXPLICIT document/file requests.
#
# The LLM classifier is occasionally unreliable for blunt requests like
# "give me a pdf document" (it sometimes answers document:false), which means
# no download card is shown. This regex layer catches the unambiguous cases and
# OVERRIDES the classifier so an explicit "...as a pdf / a word doc / an excel /
# zip the project" always produces the right file. It is deliberately
# conservative: it only fires on clear file-producing phrasing, so ordinary
# questions that merely mention a format ("how do I parse a PDF in Python")
# are left to the LLM.
# --------------------------------------------------------------------------
_FMT_MAP = {
    "pdf": "pdf",
    "word": "docx", "docx": "docx", "word document": "docx",
    "excel": "xlsx", "xlsx": "xlsx", "spreadsheet": "xlsx",
    "powerpoint": "pptx", "pptx": "pptx", "ppt": "pptx",
    "slides": "pptx", "slide deck": "pptx", "presentation": "pptx",
    "csv": "csv",
    "json": "json",
    "markdown": "md", "md": "md",
    "text file": "txt", "txt": "txt",
    "zip": "zip", "archive": "zip", "zip file": "zip",
}
# Longest keys first so "word document" / "text file" win over "word" / "text".
_FMT_ALT = "|".join(
    re.escape(k) for k in sorted(_FMT_MAP, key=len, reverse=True)
)
# "Strong" formats are inherently downloadable artifacts — a bare
# "give me a pdf / make an excel" is unambiguous. "Weak" formats (csv, json,
# md, txt) also name data shapes people ask about in code ("read a csv",
# "return json"), so for those we require explicit file/as/in-format wording
# (patterns 1–3) rather than the loose verb+format pattern (4).
_FMT_STRONG_ALT = "|".join(
    re.escape(k)
    for k in sorted(
        ("pdf", "word", "docx", "word document", "excel", "xlsx",
         "spreadsheet", "powerpoint", "pptx", "ppt", "slide deck",
         "zip", "zip file", "archive"),
        key=len,
        reverse=True,
    )
)
_GEN_VERB = (
    r"give|get|make|create|generat\w*|produc\w*|build|export\w*|download|"
    r"save|prepare|draft|provide|share|send|write|need|want|zip|compress"
)

# "in a <format>" with an article — excludes bare "word" so the idiom
# "in a word, yes" can't fire ("in a word document" is caught by pattern 2).
_FMT_ALT_NO_WORD = "|".join(
    re.escape(k) for k in sorted(_FMT_MAP, key=len, reverse=True)
    if k != "word"
)

# Patterns that unambiguously ask for a downloadable file in a named format.
_DOC_PATTERNS = [
    re.compile(rf"\bas\s+(?:an?\s+)?({_FMT_ALT})\b", re.I),
    # "in / into a <format>" — "into" so "put this into an excel spreadsheet"
    # and "turn this into a pdf" are caught (was "in"-only, which misses the
    # very common "into" phrasing).
    re.compile(rf"\b(?:in|into)\s+(?:an?\s+)?({_FMT_ALT_NO_WORD})"
               rf"(?:\s+(?:format|form))?\b", re.I),
    # "in word format/form" — the format/form suffix disambiguates from the
    # "in a word, yes" idiom excluded above.
    re.compile(r"\bin\s+(word)\s+(?:format|form)\b", re.I),
    re.compile(
        rf"\b({_FMT_ALT})\s+"
        r"(?:document|file|report|version|copy|export|sheet|spreadsheet|"
        r"workbook|presentation|deck|slideshow|archive|doc)s?\b",
        re.I,
    ),
    # "convert / turn / put / save / export … (in)to / as a <strong format>".
    # Restricted to STRONG (inherently-downloadable) formats — pdf/word/excel/
    # powerpoint/zip and their extensions — so a coding "convert the list to
    # json" or "parse this to csv" is NOT swept up as a file request. Bare
    # "word" is excluded (the "in a word" idiom); "word document"/"docx" stay.
    re.compile(
        rf"\b(?:convert|turn|put|save|export)\b[^.?!\n]{{0,30}}?"
        rf"\b(?:in|into|to|as)\s+(?:an?\s+)?"
        rf"({'|'.join(re.escape(k) for k in sorted((s for s in ('pdf', 'docx', 'word document', 'excel', 'xlsx', 'spreadsheet', 'powerpoint', 'pptx', 'ppt', 'slide deck', 'zip', 'zip file', 'archive')), key=len, reverse=True))})\b",
        re.I,
    ),
    # Loose "verb … <strong format>" — only for inherently-downloadable formats.
    re.compile(
        rf"\b(?:{_GEN_VERB})\b[^.?!\n]{{0,40}}?"
        rf"\b(?:an?\s+|the\s+|me\s+(?:an?\s+)?)?({_FMT_STRONG_ALT})\b",
        re.I,
    ),
]
# GEN-VERB + "a document" (indefinite article = produce-NEW; "summarize the
# document" refers to existing input and stays untouched) — catches "generate
# a document on kafka", "create a document about our API", "make me a document
# for onboarding". Guarded by _HOWTO_OR_LANG_RE at the call site so "how do I
# create a document in python-docx" stays a code question.
_DOC_PRODUCE_RE = re.compile(
    rf"\b(?:{_GEN_VERB})\b[^.?!\n]{{0,40}}?"
    r"\b(?:an?|another|one)\s+(?:\w+\s+)?document\b",
    re.I,
)
# Deliverable phrasings that name no format → default pdf: "as a document",
# "put this into a document", "make this downloadable", "export this
# conversation", "soft copy". Each requires a clear produce/deliver signal.
_DOC_GENERIC_RES = [
    re.compile(r"\bas\s+a\s+(?:\w+\s+)?document\b", re.I),
    re.compile(r"\b(?:put|turn|convert)\b[^.?!\n]{0,30}?"
               r"\b(?:in|into)\s+a\s+document\b", re.I),
    re.compile(r"\bmake\s+(?:this|that|it)\s+downloadable\b", re.I),
    re.compile(r"\bdownloadable\s+(?:document|file|copy|version|report)\b",
               re.I),
    re.compile(r"\bexport\s+(?:this|the|our)\s+"
               r"(?:conversation|chat|discussion|thread|answer|response)\b",
               re.I),
    re.compile(r"\bsoft\s+copy\b", re.I),
]
# Whole-project archive phrasing → zip, even without the word "zip".
_ZIP_RE = re.compile(
    r"\b(zip|compress|archive)\b|\b(whole|entire|all\s+the)\s+"
    r"(project|codebase|files?|repo)\b",
    re.I,
)
# "package/bundle the code (for download)" — verbs that mean "make me an
# archive" but are also common NOUNS ("an npm package"), so they only count
# followed by a determiner ("package the code", "bundle this up"), never as
# "a python package that…".
_PKG_RE = re.compile(
    r"\b(?:package|bundle)\s+(?:up\s+)?(?:the|this|my|everything|all)\b",
    re.I,
)
# RETRIEVAL of an already-requested document/file ("where is the document",
# "show me the file", "resend the pdf", "download the document"). This makes a
# doc-location follow-up actually re-produce the downloadable card instead of
# letting the model confabulate ("I already provided it / use the Download
# button"). Requires a retrieval verb AND a concrete document/file/format noun —
# the bare, ambiguous in-chat nouns ("report"/"summary") are deliberately
# EXCLUDED so "give me a report" stays an ordinary answer (no over-generation).
# NOTE: only RETRIEVAL/delivery verbs (no generate/create/make/open — those
# either need a named format, are handled by _DOC_PATTERNS, or mis-fire on
# coding asks like "create a file to store data" / "how do I open a pptx").
_DOC_RETRIEVAL_RE = re.compile(
    r"\b(?:where(?:'s| is| are)|show me|send me|resend|re-?send|give me|"
    r"get me|can i (?:get|have|see)|provide|share|download)\b"
    r"[^.?!\n]{0,24}?"
    r"\b(?:the|my|that|this|a|an|another)?\s*"
    r"(document|doc|file|pdf|docx?|word\s+doc(?:ument)?|excel|xlsx|spreadsheet|"
    r"powerpoint|pptx?|slide\s*deck|slides?|presentation|csv|markdown|"
    r"text\s+file|txt|json|attachment)\b",
    re.I,
)
# A programming how-to / language context turns a "download a file"-style phrase
# into a CODE question, not a document request → suppress retrieval there.
_HOWTO_OR_LANG_RE = re.compile(
    r"^\s*how\s+(?:do|to|can|would|should|might)\b|"
    r"\bin\s+(?:python|java|javascript|typescript|c\+\+|c#|golang|go|rust|ruby|"
    r"php|kotlin|swift|scala|dart|node|bash|powershell)\b|"
    r"\b(using|with)\s+(?:python|java|pandas|numpy|openpyxl|reportlab|"
    r"python-docx|fpdf|pdfkit)\b",
    re.I,
)


def explicit_doc_request(text: str) -> tuple[bool, str | None]:
    """Deterministic detection of an explicit "produce a file" request.

    Returns ``(True, canonical_format)`` for a clear request, else
    ``(False, None)`` meaning "no opinion — let the LLM classifier decide".
    """
    t = (text or "").strip().lower()
    if not t:
        return False, None
    _howto = bool(_HOWTO_OR_LANG_RE.search(t))
    for i, pat in enumerate(_DOC_PATTERNS):
        m = pat.search(t)
        if not m:
            continue
        # The LOOSE patterns (2: format+noun adjacency, 3: verb…format)
        # misfire inside coding how-tos ("how do I zip files in java",
        # "create a document in python-docx") — suppress them there. The
        # explicit as/in-format phrasings (0–1) stay live even in a how-to.
        if i >= 2 and _howto:
            continue
        # A format token inside a hyphenated library name (python-docx,
        # python-pptx, html-pdf) is code context, never a deliverable.
        if m.start(1) > 0 and t[m.start(1) - 1] == "-":
            continue
        return True, _FMT_MAP.get(m.group(1).strip(), "pdf")
    # Produce-a-document / deliverable phrasings with no named format → pdf.
    # (Named-format patterns above win, so "a word document about X" is docx.)
    if not _howto:
        if _DOC_PRODUCE_RE.search(t):
            return True, "pdf"
        for pat in _DOC_GENERIC_RES:
            if pat.search(t):
                return True, "pdf"
    # "zip the project" / "compress the whole codebase" / "package the code"
    # with no format word ("bundle this up … download" counts too). The
    # how-to guard keeps "how do I compress files in python" a code question.
    if (_ZIP_RE.search(t) or _PKG_RE.search(t)) and not _howto and (
        re.search(
            r"\b(project|codebase|repo|app|files?|folder|everything|code|"
            r"scripts?)\b", t)
        or (_PKG_RE.search(t) and "download" in t)
    ):
        return True, "zip"
    # A source-code file request → the language's extension (or "code" when the
    # language isn't named, resolved from the answer's first fenced block).
    ce = explicit_code_request(t)
    if ce:
        return True, ce
    # Retrieval of an already-requested document/file → produce it (default to
    # the format named in the noun, else pdf; the caller inherits a prior format
    # from the recent window when this turn names none). Suppressed for coding
    # how-tos ("how do I open a pptx in python") which merely name a format.
    m = _DOC_RETRIEVAL_RE.search(t)
    if m and not _HOWTO_OR_LANG_RE.search(t):
        noun = (m.group(1) or "").strip()
        return True, _FMT_MAP.get(noun, _FMT_MAP.get(noun.split()[0], "pdf"))
    # SEMANTIC TAIL (2026-07-09): dynamic understanding via exemplar
    # embeddings — the patterns above are zero-latency fast-paths; the
    # semantic gate is the authority for every phrasing they don't
    # anticipate. Fail-open: embedder warming/absent → deterministic verdict.
    if not _howto:
        sem = _semantic_doc_request(t)
        if sem is not None:
            return sem
    return False, None


def _fmt_in_text(t: str) -> str | None:
    """The canonical format named anywhere in the text (longest token wins)."""
    for key in sorted(_FMT_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", t, re.I):
            return _FMT_MAP[key]
    return None


def _semantic_doc_request(t: str) -> tuple[bool, str] | None:
    """Embedding-gate verdict for a produce-a-file ask; None = no opinion
    (embedder unavailable) so the deterministic answer stands."""
    try:
        from app.semantics import gates as _gates
        arch = _gates.matches("archive_request", t)
        if arch:
            return True, "zip"
        doc = _gates.matches("document_request", t)
        if doc:
            return True, _fmt_in_text(t) or "pdf"
        if arch is None and doc is None:
            return None                     # embedder unavailable
        return None                         # gates answered False → no doc
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Source-code file requests ("give me a python file", "this code file", …).
# Maps a language word / extension to its canonical file extension.
# --------------------------------------------------------------------------
_CODE_LANG = {
    "python": "py", "py": "py", "javascript": "js", "js": "js", "jsx": "jsx",
    "typescript": "ts", "ts": "ts", "tsx": "tsx", "java": "java",
    "kotlin": "kt", "kt": "kt", "ruby": "rb", "rb": "rb", "go": "go",
    "golang": "go", "rust": "rs", "rs": "rs", "cpp": "cpp", "c++": "cpp",
    "csharp": "cs", "c#": "cs", "cs": "cs", "php": "php", "swift": "swift",
    "scala": "scala", "dart": "dart", "kotlin script": "kt", "lua": "lua",
    "perl": "pl", "shell": "sh", "bash": "sh", "powershell": "ps1",
    "sql": "sql", "html": "html", "css": "css", "scss": "scss",
    "yaml": "yaml", "yml": "yaml", "xml": "xml", "toml": "toml",
    "graphql": "graphql", "vue": "vue", "svelte": "svelte", "c": "c",
    "go lang": "go", "objective-c": "m",
}
_CODE_EXTS = set(_CODE_LANG.values())
_CODE_WORD_ALT = "|".join(
    re.escape(w) for w in sorted(_CODE_LANG, key=len, reverse=True)
)
_CODE_EXT_ALT = "|".join(
    re.escape(e) for e in sorted(_CODE_EXTS, key=len, reverse=True)
)
# "<language> file/script" — only these nouns (others like "class" / "module"
# are ambiguous: "a python class" is a code question, not a file request).
_CODE_NAMED_RE = re.compile(
    rf"\b({_CODE_WORD_ALT})\s+(?:file|script)\b",
    re.I,
)
# "as a python file", "in rust", written-out forms.
_CODE_AS_RE = re.compile(
    rf"\bas\s+(?:an?\s+)?({_CODE_WORD_ALT})\s+(?:file|script|code|program)\b",
    re.I,
)
# ".py file", ".java file".
_CODE_DOT_RE = re.compile(rf"\.({_CODE_EXT_ALT})\b", re.I)
# Generic "code file" / "source file" with no language named.
_CODE_GENERIC_RE = re.compile(
    r"\b(?:source\s*code|code|source|script)\s+files?\b", re.I
)


def explicit_code_request(text: str) -> str | None:
    """Extension for an explicit source-code-file request, ``"code"`` when the
    language isn't named (resolved later from the answer), else ``None``."""
    t = (text or "").strip().lower()
    if not t:
        return None
    m = _CODE_NAMED_RE.search(t) or _CODE_AS_RE.search(t)
    if m:
        return _CODE_LANG.get(m.group(1).strip(), "txt")
    m = _CODE_DOT_RE.search(t)
    if m and re.search(r"\bfile\b", t):
        return m.group(1)
    # Generic "code file" — only when it's clearly a request (a generation verb
    # is present), so "the code file is broken" isn't treated as a download.
    if _CODE_GENERIC_RE.search(t) and re.search(rf"\b(?:{_GEN_VERB})\b", t):
        return "code"
    return None


def infer_code_ext(content: str) -> str:
    """Extension of the first fenced code block's language in ``content``
    (e.g. ```python → 'py'). Falls back to 'txt'."""
    m = re.search(r"```([A-Za-z0-9+#.\-]*)", content or "")
    if m:
        lang = m.group(1).strip().lower()
        if lang:
            return _CODE_LANG.get(lang, _FMT_MAP.get(lang, lang) or "txt")
    return "txt"


# Format keywords scanned ONCE a request is confirmed to be a document request,
# to support "give me a text AND a markdown document" → ["txt", "md"]. Includes
# the bare word "text" (→ txt) which is too weak to flag a request on its own
# but is fine to collect inside an already-confirmed document request.
# Bare "word" / "slides" are normal English ("in other words") and used to
# false-match; they're gone — only unambiguous tokens remain. Bare "text"
# stays because "a text and a markdown document" is a real list ask, but the
# list-separator rule below stops "…with the text of…" from adding a second
# file (the observed two-documents bug).
_MULTI_FMT = {
    "pdf": "pdf",
    "word document": "docx", "word doc": "docx", "docx": "docx",
    "excel": "xlsx", "xlsx": "xlsx", "spreadsheet": "xlsx",
    "powerpoint": "pptx", "pptx": "pptx", "ppt": "pptx",
    "slide deck": "pptx", "presentation": "pptx",
    "csv": "csv",
    "json": "json",
    "markdown": "md",
    "text file": "txt", "plain text": "txt", "text": "txt", "txt": "txt",
    "zip": "zip", "archive": "zip",
}
_MULTI_RE = re.compile(
    "|".join(
        rf"\b{re.escape(k)}\b"
        for k in sorted(_MULTI_FMT, key=len, reverse=True)
    ),
    re.I,
)


# Document formats users ask for that this app CANNOT generate. Detected so
# the answer can say "X is out of scope" + list the supported formats, instead
# of silently defaulting to PDF or hallucinating the file.
_UNSUPPORTED_FMT_RE = re.compile(
    r"\b(keynote|apple pages|\.pages|numbers file|odt|ods|odp|"
    r"open ?document|epub|mobi|rtf|latex|\.tex|one ?note|"
    r"google (?:docs?|sheets?|slides?)|indesign|publisher file|visio)\b",
    re.I,
)


def unsupported_doc_formats(text: str) -> list[str]:
    """Unsupported document formats the request names (deduped, in order)."""
    out: list[str] = []
    for m in _UNSUPPORTED_FMT_RE.finditer(text or ""):
        t = m.group(0).strip()
        if t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return out


# Multiple documents require an EXPLICIT list ("a txt AND a pdf", "pdf, docx").
# Two format words merely co-occurring ("convert my word doc to pdf") is one
# deliverable, not two.
_FMT_LIST_SEP = re.compile(r"\b(and|plus|both|as well as)\b|[,&+/]", re.I)


def explicit_doc_formats(text: str) -> list[str]:
    """All distinct file formats an explicit request names, in order.

    Returns ``[]`` when it isn't a clear document request (defer to the LLM).
    One format per request unless the user EXPLICITLY lists several: "a text
    file and a markdown document" → ``["txt", "md"]``; "give me a pdf" →
    ``["pdf"]``; "convert my word doc to pdf" → ``["pdf"]`` (not two).
    """
    det, primary = explicit_doc_request(text)
    if not det:
        return []
    low = (text or "").lower()
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _MULTI_RE.finditer(low):
        canon = _MULTI_FMT[m.group(0).lower()]
        if canon not in found:
            found.append(canon)
            spans.append((m.start(), m.end()))
    if len(found) > 1:
        # Every adjacent pair must be joined by a list separator, else this is
        # a single-deliverable request → trust the primary format.
        for i in range(1, len(spans)):
            if not _FMT_LIST_SEP.search(low[spans[i - 1][1]:spans[i][0]]):
                found = [primary or found[0]]
                break
    return found or [primary or "pdf"]


# A SHORT clarification ANSWER that names a delivery format — the chip the user
# taps when the Clarifier asks "which format?": "Format: Word (.docx)", "PDF",
# "a zip file", "Excel (.xlsx)". This is NOT a "produce a document" phrasing, so
# explicit_doc_request / explicit_doc_formats (which require a produce-verb +
# context) both miss it — hence a separate, self-contained parser. Returns the
# canonical format, or None. Bounded to short text so it can never fire on prose.
_ANSWER_FMT = {
    "pdf": "pdf",
    "word": "docx", "docx": "docx", "doc": "docx",
    "excel": "xlsx", "xlsx": "xlsx", "xls": "xlsx", "spreadsheet": "xlsx",
    "powerpoint": "pptx", "pptx": "pptx", "ppt": "pptx", "slides": "pptx",
    "presentation": "pptx",
    "csv": "csv", "json": "json",
    "markdown": "md", "md": "md",
    "txt": "txt",
    "zip": "zip", "7z": "7z", "7zip": "7z",
}
_ANSWER_FMT_RE = re.compile(
    r"\b(pdf|word|docx?|excel|xlsx?|xls|spreadsheet|powerpoint|pptx?|ppt|"
    r"slides?|presentation|csv|json|markdown|md|txt|zip|7-?zip)\b",
    re.I,
)


def format_answer(text: str) -> str | None:
    """The delivery format named in a SHORT clarification answer.

    ``"Format: Word (.docx)"`` → ``"docx"``; ``"PDF"`` → ``"pdf"``; ``"a zip"``
    → ``"zip"``. Returns ``None`` when nothing/ambiguous or the text is too long
    to be a chip answer (so it can't misfire on ordinary prose that happens to
    mention a format). Callers should only consult this on a clarification-answer
    turn (``allow_recent_doc``), never on a fresh request.
    """
    t = (text or "").strip().lower()
    if not t or len(t) > 80:
        return None
    m = _ANSWER_FMT_RE.search(t)
    if not m:
        return None
    return _ANSWER_FMT.get(m.group(1).replace("-", ""))


def mentions_format(text: str) -> bool:
    """True iff the text NAMES a concrete delivery format (pdf / word / excel /
    …) rather than a generic "document" or "file".

    explicit_doc_request / explicit_doc_formats always resolve to *some* format
    (defaulting to pdf for "put this in a document"), so they can't answer "did
    the user actually pick a format?". This can — it matches only real format
    tokens (the ``\\bdoc\\b`` boundary means "document" does NOT count). Used to
    keep the progress label honest: no "Generating PDF…" for a request that
    never said PDF.
    """
    return bool(_ANSWER_FMT_RE.search((text or "").lower()))


# --------------------------------------------------------------------------
# Agentic build/edit intent (chat) — Spec §8.6.
#
# Decide whether a chat turn should be handed to the AGENT LOOP (which reads /
# edits / builds / runs a real workspace) instead of the normal answer stream:
#   • build  — a spec doc (or existing workspace) + "build/create an app/api/…"
#   • edit   — a code archive uploaded (or a workspace already exists for the
#              conversation) + an action verb (fix/optimize/refactor/add/…)
# Read-only asks ("explain this code", "review the project", "how does X work")
# stay on the chat-mesh Q&A path — no workspace mutation.
# Deterministic + conservative; the caller may layer the LLM fallback below.
# --------------------------------------------------------------------------
_BUILD_VERBS = (
    r"build|create|generat\w*|implement|scaffold|make|develop|bootstrap|"
    r"spin\s+up|set\s+up|put\s+together|code\s+up|write"
)
_BUILD_OBJECTS = (
    r"app|application|applications|project|projects|service|services|"
    r"micro-?service|api|website|web\s*site|web\s*app|webapp|backend|"
    r"front-?end|server|program|tool|cli|bot|game|dashboard|system|"
    r"platform|prototype|mvp|library|package|crud|clone|site"
)
_EDIT_VERBS = (
    r"fix|debug|optimi[sz]e|optimi[sz]ation|refactor|enhance|enhancement|add|"
    r"implement|change|migrate|update|improve|extend|rewrite|rework|patch|"
    r"resolve|correct|repair|remove|delete|rename|integrate|upgrade|"
    r"modify|adjust|clean\s*up|finish|complete|fix\s*up|wire"
)
# A turn that OPENS with a read-only verb is a question about the code, not an
# edit request — keep it on the Q&A path even if a codebase is attached.
_READONLY_HEAD_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
    r"(explain|describe|review|summari[sz]e|analy[sz]e|walk\s+me\s+through|"
    r"tell\s+me\s+about|what\s+(?:does|is|are|happens)|how\s+(?:does|do|is|"
    r"are|can|would|should)|why\s+(?:does|is|do)|where\s+(?:is|are|does)|"
    r"which\b|who\b|show\s+me|list\b|find\b|understand|interpret|read\b)",
    re.I,
)
_BUILD_RE = re.compile(
    rf"\b(?:{_BUILD_VERBS})\b[^.?!\n]{{0,48}}?\b(?:{_BUILD_OBJECTS})\b", re.I
)
_EDIT_VERB_RE = re.compile(rf"\b(?:{_EDIT_VERBS})\b", re.I)


def detect_agentic_intent(
    text: str,
    *,
    has_archive: bool = False,
    has_spec_doc: bool = False,
    workspace_exists: bool = False,
) -> dict:
    """Route a chat turn to the agent loop or to plain Q&A (deterministic).

    Returns ``{"agentic": bool, "kind": "build"|"edit"|None,
    "workspace_required": bool}``.

    - ``has_archive``      — a code archive (zip/rar/7z/tar…) was uploaded now.
    - ``has_spec_doc``     — a spec document (pdf/docx/md/txt) was uploaded now.
    - ``workspace_exists`` — a materialized workspace already exists for this
      conversation (a prior upload), so a follow-up needs no re-upload.
    """
    res = {"agentic": False, "kind": None, "workspace_required": False}
    t = (text or "").strip()
    if not t:
        return res
    have_code = bool(has_archive or workspace_exists)
    # Read-only question about an existing codebase → stay on the Q&A path.
    if _READONLY_HEAD_RE.search(t):
        return res
    # Edit / fix / optimize over an uploaded codebase (or reused workspace).
    if have_code and _EDIT_VERB_RE.search(t):
        return {"agentic": True, "kind": "edit", "workspace_required": True}
    # Build a new app from a spec doc — or from scratch in an existing
    # workspace ("now also build a CLI for it").
    if (has_spec_doc or have_code) and _BUILD_RE.search(t):
        return {"agentic": True, "kind": "build", "workspace_required": True}
    return res


_AGENTIC_PROMPT = (
    "You route a chat turn in a coding assistant. Decide if the user wants the "
    "agent to BUILD or EDIT code in a real project workspace, versus just "
    "asking a question.\n"
    "- \"build\": create a new app/project/service from a spec.\n"
    "- \"edit\": modify an existing codebase (fix/optimize/refactor/add/…).\n"
    "- \"none\": a read-only question (explain/review/summarize/how does it "
    "work) or anything not requiring workspace changes.\n"
    "Context: a code archive was uploaded = {archive}; a spec document was "
    "uploaded = {doc}; a workspace already exists = {ws}.\n\n"
    "Reply with ONLY a compact JSON object and nothing else:\n"
    "{{\"kind\": \"build|edit|none\"}}\n\nUser message:\n"
)


async def infer_agentic_intent(
    text: str,
    *,
    has_archive: bool = False,
    has_spec_doc: bool = False,
    workspace_exists: bool = False,
) -> dict:
    """Deterministic detection first; on no-match, an LLM tie-breaker (so a
    phrasing the regex misses still routes correctly). Safe default = not
    agentic on empty input or any classifier failure."""
    det = detect_agentic_intent(
        text, has_archive=has_archive, has_spec_doc=has_spec_doc,
        workspace_exists=workspace_exists,
    )
    if det["agentic"] or not (has_archive or has_spec_doc or workspace_exists):
        return det
    # A codebase/spec is present but the deterministic rules didn't fire — ask
    # the fast classifier to break the tie.
    try:
        raw = await llm.complete(
            [{"role": "user", "content": _AGENTIC_PROMPT.format(
                archive=has_archive, doc=has_spec_doc,
                ws=workspace_exists) + (text or "")[:2000]}],
            model=(cfg.llm.classifier_model or cfg.llm.model),
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.micro_label},
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001
        log.info("agentic-intent classify failed (assuming Q&A): %s", exc)
        return det
    kind = str(_parse(raw).get("kind") or "none").lower()
    if kind in ("build", "edit"):
        return {"agentic": True, "kind": kind, "workspace_required": True}
    return det


_PROMPT = (
    "Decide whether the user's message is asking the assistant to PRODUCE a "
    "downloadable document or file — e.g. \"make a PDF report\", \"export this "
    "as Excel\", \"put it into a Word doc\", \"generate a CSV\", \"give me a "
    "ZIP of the project\", \"zip the whole project\". This is NOT a "
    "document request for ordinary questions, explanations, summaries, code, or "
    "edits that merely involve content.\n\n"
    "Reply with ONLY a compact JSON object and nothing else:\n"
    "{\"document\": true|false, \"format\": \"pdf|docx|pptx|xlsx|csv|json|md|txt|zip\"}\n"
    "When document=true, use the format the user named, defaulting to \"pdf\" "
    "if they didn't specify. Use \"zip\" when they ask for a zip / archive / "
    "compressed file or want the whole project / all files as a download. "
    "No prose, no markdown, no code fences.\n\n"
    "User message:\n"
)


def _parse(raw: str) -> dict:
    """Extract the JSON object from a model reply, tolerating fences/prose."""
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


async def infer_document_intent(text: str) -> tuple[bool, str]:
    """Return ``(wants_document, format)``. LLM-driven, provider-agnostic; safe
    default ``(False, "pdf")`` on empty input or any classifier failure.

    An explicit request ("give me a pdf", "export as Excel", "zip the project")
    is detected deterministically and short-circuits the LLM, so a blunt file
    request is never missed."""
    t = (text or "").strip()
    if not t:
        return False, "pdf"
    det, det_fmt = explicit_doc_request(t)
    if det:
        return True, det_fmt or "pdf"
    try:
        raw = await llm.complete(
            [{"role": "user", "content": _PROMPT + t[:4000]}],
            model=(cfg.llm.classifier_model or cfg.llm.model),
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.intent},
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001 — never block on this
        log.info("document-intent classify failed (assuming not a doc): %s", exc)
        return False, "pdf"
    obj = _parse(raw)
    fmt = str(obj.get("format") or "pdf").lower()
    return bool(obj.get("document")), (fmt if fmt in _FORMATS else "pdf")


__all__ = ["infer_document_intent", "explicit_doc_request", "explicit_doc_formats",
           "detect_agentic_intent", "infer_agentic_intent"]
