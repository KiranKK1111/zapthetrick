"""Universal Document & Code Transformation flow (roadmap Phase 4 #20).

Stitches the already-built pieces into ONE orchestrated turn:

    parse (any upload) → transform (inject: repair / beautify) → format
    (polyglot for code, markdown polish for docs) → validate (sandbox for code,
    structural checks for docs) → a validated result ready to download.

The transform STEP is injectable — the LLM-driven "beautify / repair" is a
runtime call, so callers pass it in; everything around it is deterministic and
offline-testable. Every stage is fail-open: a formatter or sandbox that isn't
available degrades to "not formatted / not validated", never an exception.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# ext → (language for the code formatter, language for the sandbox verifier)
_CODE_LANGS = {
    "py": "python", "js": "javascript", "ts": "typescript", "go": "go",
    "rs": "rust", "java": "java", "c": "c", "cpp": "cpp", "rb": "ruby",
    "sh": "bash", "sql": "sql",
}
_MD_EXTS = {"md", "markdown", "txt", "rst"}


@dataclass
class TransformResult:
    content: str
    kind: str                 # code | markdown | text
    language: str = ""
    ext: str = ""
    formatted: bool = False
    validated: bool = False
    report: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "language": self.language, "ext": self.ext,
            "formatted": self.formatted, "validated": self.validated,
            "report": self.report, "chars": len(self.content),
        }


def classify(content: str, filename: str = "") -> tuple[str, str]:
    """Return (kind, ext). Filename extension wins; else sniff the content."""
    ext = ""
    if "." in (filename or ""):
        ext = filename.rsplit(".", 1)[-1].lower()
    if ext in _CODE_LANGS:
        return "code", ext
    if ext in _MD_EXTS:
        return ("markdown" if ext in ("md", "markdown") else "text"), ext
    # No decisive extension → sniff.
    try:
        from app.documents.detect import infer_code_ext
        guessed = (infer_code_ext(content) or "").lstrip(".").lower()
    except Exception:  # noqa: BLE001
        guessed = ""
    if guessed in _CODE_LANGS:
        return "code", guessed
    # Markdown structure?
    import re
    if re.search(r"(^|\n)#{1,6}\s|```|(^|\n)[-*]\s|\[[^\]]+\]\([^)]+\)",
                 content or "", re.MULTILINE):
        return "markdown", "md"
    return "text", "txt"


async def transform_content(
    content: str,
    *,
    filename: str = "",
    kind: str | None = None,
    transformer: Callable[[str], Awaitable[str]] | None = None,
    do_validate: bool = True,
) -> TransformResult:
    """Run the full transform flow over in-memory [content]."""
    detected_kind, ext = classify(content or "", filename)
    kind = kind or detected_kind

    # 1. transform (LLM-driven repair/beautify, injected) — or identity.
    out = content or ""
    if transformer is not None:
        try:
            out = await transformer(out) or out
        except Exception:  # noqa: BLE001
            pass

    result = TransformResult(content=out, kind=kind, ext=ext)

    # 2. format.
    if kind == "code":
        lang = _CODE_LANGS.get(ext, "")
        result.language = lang
        if lang:
            try:
                from app.polyglot.formatters import format_code
                formatted = await format_code(lang, out)
                if formatted and formatted.strip():
                    result.content = formatted
                    result.formatted = True
            except Exception:  # noqa: BLE001
                pass
    else:
        try:
            from app.response_arch.polish import polish
            result.content = polish(out)
            result.formatted = True
        except Exception:  # noqa: BLE001
            pass

    # 3. validate.
    if do_validate and kind == "code" and result.language:
        try:
            from app.sandbox.executor import verify_script
            res = verify_script(result.content, language=result.language)
            result.validated = res.ok
            result.report = res.as_dict()
        except Exception:  # noqa: BLE001
            pass
    elif do_validate:
        # Docs: a "valid" result is simply non-empty, well-formed text.
        result.validated = bool(result.content.strip())
        result.report = {"chars": len(result.content)}

    return result


async def transform_upload(
    data: bytes,
    filename: str,
    *,
    transformer: Callable[[str], Awaitable[str]] | None = None,
    do_validate: bool = True,
) -> TransformResult:
    """Parse any supported upload to text, then run [transform_content]."""
    try:
        from app.documents.parser import extract_document_text
        text = extract_document_text(data, filename)
    except Exception:  # noqa: BLE001
        text = data.decode("utf-8", errors="replace") if data else ""
    return await transform_content(
        text, filename=filename, transformer=transformer, do_validate=do_validate)


__all__ = ["TransformResult", "classify", "transform_content", "transform_upload"]
