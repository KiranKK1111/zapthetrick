"""Diagram artifact pipeline — the IR-first half of MermaidDiagramVisualizations.md.

The doc's central argument: **don't let the model emit the final artifact**. Its
priority list, and where each item lives:

  1. Intermediate representation (AST/JSON) instead of final artifacts → :mod:`ir`
  2. Compiler + validator pipeline                                     → :mod:`validators`
  3. Universal sandbox                                                 (existing: `app/sandbox`)
  4. Automatic repair loop driven by compiler errors                   (existing: `api/routes_mermaid.py`)
  5. Streaming progress UI                                             → :mod:`stages`
  6. Versioning + targeted edits                                       → :mod:`versions`, :mod:`edits`
  7. Caching                                                           (existing: FE PNG cache + `response_arch.mermaid`)

Plus the doc's numbered improvements this package adds: post-render **quality
score** (#4), **auto-layout** planning / ELK (#5), **multiple validators** —
semantic, style, accessibility (#6, #7), **AI critic** (#15), **export
everywhere** (#16), **pipeline-stage UX** (#17).

Design rules every module here follows:
  * **deterministic core, injected model.** The IR → Mermaid emitter, the
    validators, the quality score, the edit applier and every exporter are pure
    functions with no I/O — the model only ever produces *IR JSON* or *edit ops*,
    never final syntax. That is the whole reliability argument of the doc.
  * **fail-open.** Any error degrades to "no finding" / "unchanged source", never
    an exception into a turn.
  * **round-trippable.** :func:`parse.from_mermaid` lifts existing source into the
    IR so validation, edits, versioning and export work on diagrams the model
    already wrote (and on hand-written ones).

Naming note: the flat re-exports below deliberately AVOID the names ``export``
and ``versions``. Both are submodule names, and re-exporting a function/instance
under them shadows the module on the package object, so
``import app.diagrams.export as X`` silently hands back a function. They are
re-exported as :func:`export_diagram` and :data:`version_store` instead.
"""
from __future__ import annotations

# Import the submodules first so `app.diagrams.<name>` always resolves to the
# MODULE, whatever else this package re-exports.
from app.diagrams import (  # noqa: F401
    critic, edits, export, ir, lane, layout, parse, planner, quality, stages,
    validators, versions,
)
from app.diagrams.edits import EditResult, apply_edits
from app.diagrams.export import EXPORT_FORMATS
from app.diagrams.export import export as export_diagram
from app.diagrams.ir import (
    DiagramIR, Edge, Group, Node, from_dict, to_mermaid,
)
from app.diagrams.lane import LaneResult, compile_answer, compile_diagrams
from app.diagrams.layout import (
    LayoutPlan, count_crossings, order_nodes, plan_layout, to_elk_json,
)
from app.diagrams.parse import from_mermaid
from app.diagrams.quality import QualityScore, score
from app.diagrams.stages import STAGES, Stage, StageTracker, stage_ladder
from app.diagrams.validators import Finding, validate, validate_source
from app.diagrams.versions import DiagramVersion
from app.diagrams.versions import versions as version_store

__all__ = [
    # submodules
    "critic", "edits", "export", "ir", "lane", "layout", "parse", "planner",
    "quality", "stages", "validators", "versions",
    # flat conveniences
    "DiagramIR", "Node", "Edge", "Group", "from_dict", "to_mermaid",
    "from_mermaid",
    "Finding", "validate", "validate_source",
    "QualityScore", "score",
    "LaneResult", "compile_diagrams", "compile_answer",
    "LayoutPlan", "plan_layout", "to_elk_json", "order_nodes", "count_crossings",
    "EditResult", "apply_edits",
    "DiagramVersion", "version_store",
    "EXPORT_FORMATS", "export_diagram",
    "STAGES", "Stage", "StageTracker", "stage_ladder",
]
