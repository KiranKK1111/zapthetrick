"""ResponseOrchestrator — the coherent named umbrella (roadmap Phase 6 #3).

The finalize / envelope / verify / stream path was real but *distributed*: the
streaming generator in `routes_agents` reached into `layer.finalize`,
`envelope.build_envelope`, the block/plan/analytics/budget helpers, and the
quality controller ad hoc, with no single object owning the turn. This is that
object — one named orchestrator that sequences a streaming turn end to end:

    plan ─► stream-mode ─► [tokens ─► blocks/artifacts + analytics] ─►
    finalize (shape) ─► envelope

It does not replace the route; it is the umbrella the route (or another agent)
drives, so the pipeline has ONE coherent surface instead of scattered calls.
Every step is fail-open — a failure in any stage degrades to passthrough and
never breaks the stream.

Typical use inside a streaming generator::

    orch = ResponseOrchestrator(question, intent=intent, shape=shape, depth=depth)
    yield _sse(*orch.plan_frame())                 # first meaningful paint (#5)
    async for delta in model_stream:
        yield _sse("token", {"text": delta})
        for ev, data in orch.on_token(delta):      # block/artifact frames (#6/#18)
            yield _sse(ev, data)
    for ev, data in orch.flush():
        yield _sse(ev, data)
    shaped = orch.finalize(full_text)              # shape/polish (#4 post-pass)
    yield _sse(*orch.analytics_frame())            # TTFMU/first-code/... (#21)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import analytics as _an
from .blocks import Block, BlockAssembler
from .budget import StreamBudget, load_budget
from .content_router import Shape
from .layer import ShapedResponse, finalize
from .plan import ResponsePlan, build_response_plan
from .stream_shape import StreamMode, stream_mode_for


@dataclass
class ResponseOrchestrator:
    question: str = ""
    intent: str | None = None
    shape: Shape | str | None = None
    depth: str = "standard"

    plan: ResponsePlan = field(init=False)
    stream_mode: StreamMode = field(init=False)
    budget: StreamBudget = field(init=False)
    analytics: "_an.StreamAnalytics" = field(init=False)
    assembler: BlockAssembler = field(init=False)
    _first_seen: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.plan = build_response_plan(
            self.question, intent=self.intent, shape=self.shape,
            depth=self.depth)
        self.stream_mode = stream_mode_for(self.plan.shape)
        try:
            self.budget = load_budget()
        except Exception:  # noqa: BLE001
            self.budget = StreamBudget(0.8, 5.0, 300.0)
        self.analytics = _an.StreamAnalytics().start()
        self.assembler = BlockAssembler(
            emit_artifacts=self.stream_mode.emit_artifacts)

    # -- pre-token ----------------------------------------------------------
    def plan_frame(self) -> tuple[str, dict]:
        """The ``plan`` SSE frame (upcoming sections + stream mode), pre-token."""
        data = self.plan.as_frame()
        data["stream"] = self.stream_mode.as_frame()
        return "plan", data

    # -- per-token ----------------------------------------------------------
    def on_token(self, delta: str) -> list[tuple[str, dict]]:
        """Feed a token delta; return any structured frames it completed.

        Emits ``block`` frames as logical blocks close, and ``artifact`` frames
        as file-ish code blocks land (progressive delivery, #18). The caller
        still streams the raw ``token`` itself — these are additive.
        """
        if not self._first_seen and (delta or "").strip():
            self._first_seen = True
            self.analytics.mark(_an.FIRST_MEANINGFUL)
        frames: list[tuple[str, dict]] = []
        if not self.stream_mode.block_aware:
            return frames
        try:
            for blk in self.assembler.feed(delta):
                frames.extend(self._block_frames(blk))
        except Exception:  # noqa: BLE001
            return frames
        return frames

    def flush(self) -> list[tuple[str, dict]]:
        """Emit the trailing open block + any final artifacts."""
        frames: list[tuple[str, dict]] = []
        if not self.stream_mode.block_aware:
            return frames
        try:
            for blk in self.assembler.flush():
                frames.extend(self._block_frames(blk))
        except Exception:  # noqa: BLE001
            pass
        return frames

    def _block_frames(self, blk: Block) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = [("block", blk.as_frame())]
        if blk.type == "code":
            self.analytics.mark(_an.FIRST_CODE)
        # Drain any artifacts the assembler surfaced on this block's close.
        while len(self.assembler.artifacts) > getattr(self, "_art_emitted", 0):
            i = getattr(self, "_art_emitted", 0)
            a = self.assembler.artifacts[i]
            self._art_emitted = i + 1
            self.analytics.mark(_an.ARTIFACT_READY)
            out.append(("artifact", {
                "filename": a.filename, "language": a.language,
                "content": a.content, "progressive": True}))
        return out

    # -- post-stream --------------------------------------------------------
    def finalize(self, full_text: str) -> ShapedResponse:
        """Run the shaping pipeline with this turn's planned shape."""
        try:
            return finalize(full_text, question=self.question,
                            shape=self.plan.shape, depth=self.depth)
        except Exception:  # noqa: BLE001
            return ShapedResponse(text=full_text or "", shape=self.plan.shape,
                                  depth=self.depth)

    def analytics_frame(self) -> tuple[str, dict]:
        """Terminal ``analytics`` telemetry frame (TTFMU / first-code / ...)."""
        self.analytics.mark(_an.DONE)
        return "analytics", self.analytics.metrics()

    def envelope(self, **kwargs) -> dict:
        """Build the response.v1 envelope with this turn's planned shape filled in."""
        from .envelope import build_envelope
        kwargs.setdefault("answer_shape", self.plan.shape.value)
        try:
            return build_envelope(**kwargs)
        except Exception:  # noqa: BLE001
            return {}


__all__ = ["ResponseOrchestrator"]
