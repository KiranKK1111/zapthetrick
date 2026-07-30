"""The answer-path compile lane — MermaidDiagramVisualizations.md #1, live.

The doc's headline argument is that the model should not be the thing that emits
final diagram syntax. `app/diagrams/ir.py` provides the deterministic generator
and `/api/diagram/compose` provides the IR-first *generation* entry point, but a
normal chat turn still gets its diagram from the model's ```mermaid``` fence. This
module puts the generator on that path:

    answer text
        │
        ▼  for each ```mermaid fence
    lift → IR  ──(refused)──▶ leave the fence exactly as written
        │
        ▼ (accepted)
    validate → plan layout → RE-EMIT deterministically
        │
        ▼
    answer text with a generator-produced fence

The result is that a diagram the model wrote is *recompiled* from its structure:
labels get quoted, `subgraph`/`end` is balanced by construction, arrow spellings
become legal, direction/spacing come from the layout planner. No model call, so it
costs nothing per turn and cannot change the prose around it.

## The safety gate (why this is not "auto-normalize everything")

Re-emitting from a partial understanding would DESTROY diagrams, so a fence is
rewritten only when every one of these holds:

  1. the flag `response_arch.diagram_ir_lane` is on;
  2. the diagram type is one the IR models — gantt, pie, journey, timeline,
     gitGraph, C4, sankey, xychart, quadrant, block, radar, treemap, kanban and
     the rest are recognised and REFUSED (`parse.unsupported_kind`);
  3. the lift understood every meaningful line (`meta["unparsed"]` is empty), so
     nothing is dropped;
  4. the source carries no author-set `%%{init}%%`, `style`, `classDef`,
     `linkStyle` or `click` directive — those are deliberate and the IR does not
     round-trip them (3 catches most of these; 4 is belt-and-braces);
  5. the emitted source, lifted AGAIN, has the same node ids and the same edge
     count — a round-trip proof that the rewrite is structure-preserving;
  6. the rewrite does not make validation worse (no new error findings).

If any check fails the original text is returned untouched. That is the whole
design: the lane can only ever improve a diagram it fully understands.

Pure and synchronous — it is a compiler pass, not a model call. Never raises.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from app.diagrams import quality as Q
from app.diagrams import validators as V
from app.diagrams.ir import DiagramIR
from app.diagrams.layout import render_with_layout
from app.diagrams.parse import from_mermaid, unsupported_kind

# A complete ```mermaid fence. Deliberately requires the CLOSING fence, so a
# half-streamed diagram is never touched.
_FENCE = re.compile(r"```mermaid[ \t]*\r?\n(.*?)```", re.S | re.I)
# Author-set presentation the IR does not model. These land in `unparsed` anyway;
# matching them explicitly makes the refusal reason legible instead of a generic
# "didn't understand".
_AUTHORED_STYLE = re.compile(
    r"^\s*(?:style\s+\S|classDef\s+\S|linkStyle\s+\S|click\s+\S)", re.M)
_INIT_DIRECTIVE = re.compile(r"^\s*%%\{\s*init\s*:\s*(.*?)\}%%\s*$", re.M | re.S)
# The ONLY keys the layout planner puts in its own `%%{init}%%`. An init directive
# whose keys are a subset of these is ours; anything else (theme, themeVariables,
# securityLevel, a per-diagram config…) was set deliberately and must be respected.
#
# This distinction matters more than it looks: without it the lane refuses its OWN
# output, because `render_with_layout` prepends a directive. That made the lane
# a one-shot (a re-saved message could never be recompiled by an improved
# generator) and made "always" planner mode a no-op.
_OUR_INIT_KEYS = frozenset({"flowchart", "layout", "elk"})
_OUR_FLOWCHART_KEYS = frozenset({
    "curve", "nodeSpacing", "rankSpacing", "useMaxWidth", "htmlLabels"})


def _authored_presentation(source: str) -> str:
    """The author-set styling in `source` that the IR does not round-trip, or ""."""
    match = _AUTHORED_STYLE.search(source)
    if match:
        return match.group(0).strip().split()[0]
    for directive in _INIT_DIRECTIVE.finditer(source):
        if not _is_our_init(directive.group(1)):
            return "%%{init}%%"
    return ""


def _is_our_init(payload: str) -> bool:
    """Was this `%%{init: …}%%` body emitted by our own layout planner?"""
    import json
    try:
        config = json.loads(payload)
    except Exception:  # noqa: BLE001 — unparseable → treat as the author's
        return False
    if not isinstance(config, dict) or not set(config) <= _OUR_INIT_KEYS:
        return False
    flowchart = config.get("flowchart")
    if flowchart is not None:
        if not isinstance(flowchart, dict):
            return False
        if not set(flowchart) <= _OUR_FLOWCHART_KEYS:
            return False
    return True


def enabled() -> bool:
    """Flag-gated. Default ON: the gate above means the lane either improves a
    diagram or leaves it alone, and the reliability win is the doc's #1 item."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.response_arch, "diagram_ir_lane", True))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class DiagramOutcome:
    """What the lane did with one fence, and why."""
    index: int
    rewritten: bool
    kind: str = ""
    reason: str = ""              # why it was left alone (empty when rewritten)
    nodes: int = 0
    edges: int = 0
    score: float | None = None
    errors_before: int = 0
    errors_after: int = 0

    def to_dict(self) -> dict:
        return {"index": self.index, "rewritten": self.rewritten,
                "kind": self.kind, "reason": self.reason, "nodes": self.nodes,
                "edges": self.edges, "score": self.score,
                "errors_before": self.errors_before,
                "errors_after": self.errors_after}


@dataclass
class LaneResult:
    text: str
    outcomes: list[DiagramOutcome] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(outcome.rewritten for outcome in self.outcomes)

    @property
    def diagrams(self) -> int:
        return len(self.outcomes)

    def to_dict(self) -> dict:
        return {"changed": self.changed, "diagrams": self.diagrams,
                "outcomes": [o.to_dict() for o in self.outcomes]}


def _round_trip_matches(original: DiagramIR, emitted: str) -> bool:
    """Does re-reading the emitted source give back the same structure?

    This is the proof that the rewrite preserved content. Node IDS must match
    exactly; edge COUNT must match (labels/styles are normalised by design, and
    comparing them would reject legitimate normalisation).
    """
    again = from_mermaid(emitted)
    if again.meta.get("unparsed"):
        return False
    return (again.node_ids == original.node_ids
            and len(again.edges) == len(original.edges)
            and {group.id for group in again.groups}
            == {group.id for group in original.groups})


def _compile_one(source: str, index: int) -> tuple[str, DiagramOutcome]:
    """→ (source_to_use, outcome). Returns the ORIGINAL source on any refusal."""
    refused = unsupported_kind(source)
    if refused:
        return source, DiagramOutcome(
            index=index, rewritten=False, kind=refused,
            reason=f"`{refused}` is not modelled by the diagram IR")

    authored = _authored_presentation(source)
    if authored:
        return source, DiagramOutcome(
            index=index, rewritten=False,
            reason=f"carries an author-set `{authored}` directive")

    ir = from_mermaid(source)
    if not ir.nodes:
        return source, DiagramOutcome(
            index=index, rewritten=False, kind=ir.kind,
            reason="nothing recognisable to rebuild")

    unparsed = ir.meta.get("unparsed") or []
    if unparsed:
        return source, DiagramOutcome(
            index=index, rewritten=False, kind=ir.kind,
            nodes=len(ir.nodes), edges=len(ir.edges),
            reason=f"{len(unparsed)} line(s) not understood: {unparsed[0][:60]}")

    before = V.validate(ir, source=source)
    emitted, _plan = render_with_layout(ir)

    if not _round_trip_matches(ir, emitted):
        return source, DiagramOutcome(
            index=index, rewritten=False, kind=ir.kind,
            nodes=len(ir.nodes), edges=len(ir.edges),
            reason="the rebuild did not round-trip identically")

    after = V.validate(from_mermaid(emitted))
    errors_before = len(before.errors)
    errors_after = len(after.errors)
    if errors_after > errors_before:
        return source, DiagramOutcome(
            index=index, rewritten=False, kind=ir.kind,
            nodes=len(ir.nodes), edges=len(ir.edges),
            errors_before=errors_before, errors_after=errors_after,
            reason="the rebuild would introduce new problems")

    score = Q.score_findings(after.findings)
    return emitted, DiagramOutcome(
        index=index, rewritten=True, kind=ir.kind,
        nodes=len(ir.nodes), edges=len(ir.edges), score=score.overall,
        errors_before=errors_before, errors_after=errors_after)


def compile_diagrams(text: str) -> LaneResult:
    """Recompile every eligible ```mermaid``` fence in `text` from its structure.

    Fail-open in every direction: disabled, no fences, an unmodelled diagram type,
    an incomplete lift or any exception → the original text, unchanged.
    """
    if not text or "```mermaid" not in text.lower():
        return LaneResult(text=text or "")
    if not enabled():
        return LaneResult(text=text)
    try:
        outcomes: list[DiagramOutcome] = []
        pieces: list[str] = []
        cursor = 0
        for index, match in enumerate(_FENCE.finditer(text)):
            pieces.append(text[cursor:match.start()])
            body = match.group(1)
            replacement, outcome = _compile_one(body.strip(), index)
            outcomes.append(outcome)
            if outcome.rewritten:
                pieces.append(f"```mermaid\n{replacement}\n```")
            else:
                pieces.append(match.group(0))
            cursor = match.end()
        pieces.append(text[cursor:])
        return LaneResult(text="".join(pieces), outcomes=outcomes)
    except Exception:  # noqa: BLE001 — a compiler pass must never break a turn
        return LaneResult(text=text)


def compile_answer(text: str) -> str:
    """The one-liner the answer paths call: recompiled text, or `text` unchanged."""
    return compile_diagrams(text).text


# ==========================================================================
# The PLANNER lane — MermaidDiagramVisualizations.md #1, the model half
# ==========================================================================
# `compile_diagrams` above closes the doc's reliability argument for SYNTAX: the
# generator, not the model, emits the final Mermaid. What it cannot fix is a
# diagram whose STRUCTURE the reader can't fully lift (so the compile is refused)
# or one that parses but is wrong. For those, the doc's actual prescription is:
#
#     Prompt → Planner → Diagram IR (JSON) → Mermaid Generator → Parser → SVG
#
# i.e. re-derive the diagram as STRUCTURE and generate from that. This lane does
# it, and it is off by default because it costs a model round trip.
#
# ## Modes (`cfg.response_arch.diagram_planner`)
#   "off"     — never call the planner. The deterministic lane still runs.
#   "repair"  — call it only when the deterministic path could NOT produce a clean
#               diagram: the lift was incomplete, or the compiled diagram still has
#               validator errors, or its quality score is below the floor. This is
#               the recommended setting: a model call is spent only on the diagrams
#               that were actually broken.
#   "always"  — re-plan every modelled diagram from structure, whatever the model
#               wrote. The doc's purest reading of #1, at one extra call per diagram.
#
# ## The acceptance gate
# A planned diagram REPLACES the existing one only if it is measurably better:
# no validator errors, a score at least as high, and no loss of content (it must
# keep at least as many nodes and edges). A planner that hallucinates a smaller
# diagram loses; the deterministic result stands. Without that gate, turning the
# flag on could downgrade diagrams the model got right.

PLANNER_OFF = "off"
PLANNER_REPAIR = "repair"
PLANNER_ALWAYS = "always"
PLANNER_MODES = (PLANNER_OFF, PLANNER_REPAIR, PLANNER_ALWAYS)

# Below this score a compiled diagram is considered worth re-planning in "repair"
# mode. Matches `quality.PASS_THRESHOLD` — the same bar the score itself uses.
PLANNER_SCORE_FLOOR = Q.PASS_THRESHOLD

# ---- budgets ------------------------------------------------------------
# These exist because of WHERE this lane runs: after the answer has finished
# streaming but BEFORE the `done` frame. The client is sitting there waiting, so
# an unbounded model call would turn a slow route into a turn that looks hung.
#
# Per-diagram wall clock. A planner call that takes longer than this is abandoned
# and the existing diagram stands — a diagram is an enhancement, never worth
# holding a turn open for.
PLANNER_TIMEOUT_S = 25.0
# Diagrams planned per answer. An answer with eight diagrams in "always" mode
# would otherwise cost eight sequential model calls; the rest keep their
# deterministically-compiled form, which is already correct syntax.
PLANNER_MAX_DIAGRAMS = 3


def planner_mode() -> str:
    """The configured planner mode. Default OFF (it costs a model call)."""
    try:
        from app.core.config_loader import cfg
        mode = str(getattr(cfg.response_arch, "diagram_planner", PLANNER_OFF) or
                   PLANNER_OFF).strip().lower()
        return mode if mode in PLANNER_MODES else PLANNER_OFF
    except Exception:  # noqa: BLE001
        return PLANNER_OFF


_REBUILD_REQUEST = (
    "Rebuild the diagram described in the context below as structure. Preserve "
    "every component and every relationship it shows — do not simplify it, and do "
    "not add anything that is not there."
)


def _needs_planning(outcome: DiagramOutcome, mode: str) -> str:
    """Why this fence should go to the planner, or "" to leave it alone."""
    # Never plan a diagram type the IR cannot emit — there would be nothing to
    # generate from. Same for one whose styling the author set deliberately.
    if outcome.reason.endswith("modelled by the diagram IR"):
        return ""
    if "author-set" in outcome.reason:
        return ""
    if mode == PLANNER_ALWAYS:
        return "planner mode is `always`"
    if mode != PLANNER_REPAIR:
        return ""
    if not outcome.rewritten:
        # The deterministic path gave up — this is exactly the case the planner
        # exists for: re-derive the structure instead of trusting the text.
        return f"deterministic rebuild refused ({outcome.reason})"
    if outcome.errors_after > 0:
        return f"{outcome.errors_after} validator error(s) remain"
    if outcome.score is not None and outcome.score < PLANNER_SCORE_FLOOR:
        return f"quality {outcome.score:.0f} below {PLANNER_SCORE_FLOOR:.0f}"
    return ""


@dataclass
class PlannedOutcome:
    """What the planner lane did with one fence."""
    index: int
    planned: bool
    trigger: str = ""             # why the planner was called
    reason: str = ""              # why its result was NOT used
    nodes_before: int = 0
    nodes_after: int = 0
    score_before: float | None = None
    score_after: float | None = None

    def to_dict(self) -> dict:
        return {"index": self.index, "planned": self.planned,
                "trigger": self.trigger, "reason": self.reason,
                "nodes_before": self.nodes_before, "nodes_after": self.nodes_after,
                "score_before": self.score_before, "score_after": self.score_after}


@dataclass
class PlannedLaneResult:
    text: str
    outcomes: list[PlannedOutcome] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(outcome.planned for outcome in self.outcomes)

    def to_dict(self) -> dict:
        return {"changed": self.changed,
                "outcomes": [o.to_dict() for o in self.outcomes]}


def _better(candidate: DiagramIR, *, nodes_before: int, edges_before: int,
            score_before: float | None) -> tuple[bool, str]:
    """Is the planned diagram an improvement worth adopting?"""
    report = V.validate(candidate)
    errors = len(report.errors)
    if errors:
        return False, f"planned diagram has {errors} validator error(s)"
    if len(candidate.nodes) < nodes_before:
        return False, (f"planned diagram lost content "
                       f"({nodes_before} → {len(candidate.nodes)} nodes)")
    if len(candidate.edges) < edges_before:
        return False, (f"planned diagram lost relationships "
                       f"({edges_before} → {len(candidate.edges)} edges)")
    score = Q.score_findings(report.findings).overall
    if score_before is not None and score + 0.5 < score_before:
        return False, (f"planned diagram scores lower "
                       f"({score:.0f} < {score_before:.0f})")
    return True, ""


async def plan_diagrams(text: str, *, request: str = "") -> PlannedLaneResult:
    """Re-derive eligible diagrams as IR via the planner, then generate from it.

    Assumes `compile_diagrams` has ALREADY run on `text` (the answer paths call
    them in that order), so what this sees is the best the deterministic path could
    do. Fail-open everywhere: mode off, no fences, no model, a planner refusal or a
    result that isn't better all return the text unchanged.
    """
    if not text or "```mermaid" not in text.lower():
        return PlannedLaneResult(text=text or "")
    mode = planner_mode()
    if mode == PLANNER_OFF:
        return PlannedLaneResult(text=text)
    try:
        from app.diagrams.planner import plan as plan_ir
    except Exception:  # noqa: BLE001
        return PlannedLaneResult(text=text)

    try:
        outcomes: list[PlannedOutcome] = []
        pieces: list[str] = []
        cursor = 0
        budget = PLANNER_MAX_DIAGRAMS
        for index, match in enumerate(_FENCE.finditer(text)):
            pieces.append(text[cursor:match.start()])
            cursor = match.end()
            source = match.group(1).strip()

            # Re-derive the deterministic verdict for THIS fence so the decision
            # and the comparison baseline come from the same place.
            _compiled, outcome = _compile_one(source, index)
            trigger = _needs_planning(outcome, mode)
            if not trigger:
                pieces.append(match.group(0))
                outcomes.append(PlannedOutcome(
                    index=index, planned=False,
                    reason=outcome.reason or "already clean"))
                continue
            if budget <= 0:
                pieces.append(match.group(0))
                outcomes.append(PlannedOutcome(
                    index=index, planned=False, trigger=trigger,
                    reason=f"planner budget spent ({PLANNER_MAX_DIAGRAMS} "
                           f"diagram(s) per answer)"))
                continue
            budget -= 1

            existing = from_mermaid(source)
            nodes_before, edges_before = len(existing.nodes), len(existing.edges)
            score_before = outcome.score
            try:
                # Bounded: this runs between the last token and the `done` frame,
                # so a slow route must not hold the turn open.
                planned, errors = await asyncio.wait_for(
                    plan_ir(
                        request.strip() or _REBUILD_REQUEST,
                        kind=existing.kind if existing.nodes else "",
                        context=f"The current diagram source is:\n{source[:4000]}",
                    ),
                    timeout=PLANNER_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, TimeoutError):
                pieces.append(match.group(0))
                outcomes.append(PlannedOutcome(
                    index=index, planned=False, trigger=trigger,
                    nodes_before=nodes_before, score_before=score_before,
                    reason=f"the planner took longer than "
                           f"{PLANNER_TIMEOUT_S:.0f}s"))
                continue
            if planned is None:
                pieces.append(match.group(0))
                outcomes.append(PlannedOutcome(
                    index=index, planned=False, trigger=trigger,
                    nodes_before=nodes_before, score_before=score_before,
                    reason="; ".join(errors[:2]) or "the planner returned nothing"))
                continue

            ok, why = _better(planned, nodes_before=nodes_before,
                              edges_before=edges_before, score_before=score_before)
            if not ok:
                pieces.append(match.group(0))
                outcomes.append(PlannedOutcome(
                    index=index, planned=False, trigger=trigger,
                    nodes_before=nodes_before, nodes_after=len(planned.nodes),
                    score_before=score_before, reason=why))
                continue

            emitted, _plan = render_with_layout(planned)
            after = Q.score_findings(V.validate(planned).findings)
            pieces.append(f"```mermaid\n{emitted}\n```")
            outcomes.append(PlannedOutcome(
                index=index, planned=True, trigger=trigger,
                nodes_before=nodes_before, nodes_after=len(planned.nodes),
                score_before=score_before, score_after=after.overall))
        pieces.append(text[cursor:])
        return PlannedLaneResult(text="".join(pieces), outcomes=outcomes)
    except Exception:  # noqa: BLE001 — never break a turn over a diagram
        return PlannedLaneResult(text=text)


async def plan_answer(text: str, *, request: str = "") -> str:
    """The one-liner the answer paths call after `compile_answer`.

    Returns immediately (no awaits, no imports) when the mode is off, so the
    default configuration pays nothing for the call site existing.
    """
    if not text or "```mermaid" not in text.lower():
        return text or ""
    if planner_mode() == PLANNER_OFF:
        return text
    return (await plan_diagrams(text, request=request)).text


__all__ = ["enabled", "DiagramOutcome", "LaneResult", "compile_diagrams",
           "compile_answer", "PLANNER_OFF", "PLANNER_REPAIR", "PLANNER_ALWAYS",
           "PLANNER_MODES", "PLANNER_SCORE_FLOOR", "PLANNER_TIMEOUT_S",
           "PLANNER_MAX_DIAGRAMS", "planner_mode",
           "PlannedOutcome", "PlannedLaneResult", "plan_diagrams", "plan_answer"]
