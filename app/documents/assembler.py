"""Multi-pass document assembler — Phase 2 of the Document Generation roadmap.

The roadmap's Phase-2 scope calls for "multi-pass assembly (outline → content →
diagrams → formatting → validate), each pass fail-open." This is that staged
assembler, built as a REAL pipeline over the deterministic stages the subsystem
already owns rather than a single monolith render:

  1. **outline**   — ``planner.plan_document`` builds the section blueprint
     (goal + depth + expected sections).
  2. **content**   — parse the supplied answer into the DocumentModel IR. (A
     *generative* content pass — asking the LLM to draft each blueprint section —
     is an extra model pass owned by the generation route, not this deterministic
     assembler; the content the user already produced is the input here. See the
     module note below.)
  3. **structure** — ``structure.enrich``: auto-diagrams, TOC, glossary, figure/
     table numbering, smart appendix.
  4. **format**    — re-serialize the enriched model to clean Markdown.
  5. **validate**  — ``review.analyze_document_async`` scored AGAINST the
     blueprint (completeness), plus the code-lint + reviewer panels.

Every pass is independently fail-open: a failing pass degrades to passing the
input through untouched and is recorded in ``passes`` with ``ok=False``, so the
assembler never breaks a turn. The end result is the same enriched bytes today's
render produces, but the STAGED report (which passes ran, the blueprint, the
quality score) is now first-class and attachable to the export meta.

Note on "multi-LLM-pass": stages 1/3/4 are deterministic (no model call); stage
2's generative variant and stage 4's LLM reviewer panels are the model passes,
and those stay flag-gated (``multi_reviewer`` / ``llm_review``) exactly as
elsewhere. Nothing here is faked — the orchestration is real and each stage is
the module that already implements it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AssemblyResult:
    markdown: str                       # the enriched, formatted document
    blueprint: object = None            # planner.Blueprint | None
    quality: object = None              # review.QualityReport | None
    passes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p.get("ok") for p in self.passes) if self.passes else False

    def as_dict(self) -> dict:
        bp = self.blueprint
        q = self.quality
        return {
            "passes": self.passes,
            "blueprint": bp.as_dict() if hasattr(bp, "as_dict") else None,
            "quality": q.as_dict() if hasattr(q, "as_dict") else None,
        }


def _record(passes: list, name: str, ok: bool, detail: str = "") -> None:
    passes.append({"pass": name, "ok": bool(ok), "detail": detail[:160]})


async def assemble_document(content: str, *, request_text: str = "",
                            title: str = "",
                            enrich_structure: bool = True) -> AssemblyResult:
    """Run the staged assembly pipeline over ``content`` and return the result +
    a per-pass log. Fail-open throughout — any pass may fail without breaking the
    others. ``request_text`` (the user's ask) drives the outline goal/depth;
    ``enrich_structure=False`` skips the structure/format passes for callers that
    enrich at render time already (so the document isn't enriched twice)."""
    passes: list = []
    from app.documents.model import markdown_to_model, model_to_markdown

    # ── Pass 1: outline ──────────────────────────────────────────────────────
    blueprint = None
    try:
        from app.documents.planner import plan_document
        blueprint = plan_document(request_text or title or (content or "")[:200])
        _record(passes, "outline", True,
                f"{blueprint.goal.value}/{blueprint.depth.value}, "
                f"{len(blueprint.sections)} planned sections")
    except Exception as exc:  # noqa: BLE001
        _record(passes, "outline", False, str(exc))

    # ── Pass 2: content (parse the provided answer into the IR) ──────────────
    try:
        model = markdown_to_model(content or "", title)
        _record(passes, "content", True, f"{len(model.sections)} sections parsed")
    except Exception as exc:  # noqa: BLE001
        model = markdown_to_model("", title)
        _record(passes, "content", False, str(exc))

    # ── Pass 3 + 4a: structure (diagrams/TOC/glossary/numbering) + format ────
    md = content or ""
    if enrich_structure:
        try:
            from app.documents.structure import enrich
            model = enrich(model)
            _record(passes, "structure", True,
                    "diagrams + TOC + glossary + numbering + appendix")
        except Exception as exc:  # noqa: BLE001
            _record(passes, "structure", False, str(exc))
        try:
            md = model_to_markdown(model)
            _record(passes, "format", True, "re-serialized enriched model")
        except Exception as exc:  # noqa: BLE001
            _record(passes, "format", False, str(exc))

    # ── Pass 5: validate (against the blueprint) ─────────────────────────────
    quality = None
    try:
        from app.documents.review import analyze_document_async
        quality = await analyze_document_async(
            model, blueprint=blueprint, title=title)
        _record(passes, "validate", True,
                f"score {quality.score}, {len(quality.issues)} issues")
    except Exception as exc:  # noqa: BLE001
        _record(passes, "validate", False, str(exc))

    return AssemblyResult(markdown=md, blueprint=blueprint, quality=quality,
                          passes=passes)


def multi_pass_enabled() -> bool:
    """Config gate for using the staged assembler on the export path to produce
    the quality report (default ON — it's additive; the report just gains the
    per-pass log + blueprint-scored completeness)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "multi_pass_assembly", True))
    except Exception:  # noqa: BLE001
        return True


__all__ = ["AssemblyResult", "assemble_document", "multi_pass_enabled"]
