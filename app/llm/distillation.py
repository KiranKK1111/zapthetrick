"""Distillation-data exporter (roadmap Phase 5 #14).

Genuine, minimal version of "distil to a local model": the missing piece for
distillation is a clean supervised dataset of (prompt → high-quality completion)
pairs harvested from real traffic. This module turns request/response traces
into a filtered, deduplicated JSONL dataset in the standard chat-fine-tuning
shape (`{"messages": [...]}` per line) that llama-factory / axolotl / the OpenAI
fine-tune format all accept.

It does NOT train anything (there is no in-repo local training runtime) — it
produces the DATA a distillation/fine-tune run consumes, with quality gates so
the student learns from good teacher outputs only: drop errors/refusals, empty or
too-short completions, and near-duplicate prompts.

Deterministic, dependency-free, fail-open.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_REFUSAL = re.compile(
    r"\b(i can'?t help with that|i cannot help with that|as an ai(?: language "
    r"model)?|i'?m unable to assist|i'?m sorry, but i can'?t)\b", re.IGNORECASE)
_ERROR = re.compile(r"\b(traceback|exception|internal server error|null null)\b",
                    re.IGNORECASE)


@dataclass
class Trace:
    """One teacher interaction eligible for the distillation set."""
    prompt: str
    completion: str
    system: str = ""
    quality: float = 1.0          # 0..1 teacher-quality signal (verifier/feedback)
    model: str = ""               # teacher model id (provenance)


@dataclass
class ExportStats:
    seen: int = 0
    kept: int = 0
    dropped_refusal: int = 0
    dropped_error: int = 0
    dropped_short: int = 0
    dropped_lowq: int = 0
    dropped_dup: int = 0
    reasons: list[str] = field(default_factory=list)


class DistillationExporter:
    """Filters + dedupes teacher traces into a chat-fine-tune JSONL dataset."""

    def __init__(self, *, min_completion_chars: int = 40,
                 min_quality: float = 0.6) -> None:
        self.min_completion_chars = min_completion_chars
        self.min_quality = min_quality

    def _prompt_key(self, system: str, prompt: str) -> str:
        norm = re.sub(r"\s+", " ", f"{system}\n{prompt}").strip().lower()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def _accept(self, t: Trace, stats: ExportStats) -> bool:
        comp = (t.completion or "").strip()
        if _REFUSAL.search(comp):
            stats.dropped_refusal += 1
            return False
        if _ERROR.search(comp):
            stats.dropped_error += 1
            return False
        if len(comp) < self.min_completion_chars:
            stats.dropped_short += 1
            return False
        if float(t.quality or 0.0) < self.min_quality:
            stats.dropped_lowq += 1
            return False
        return True

    def build(self, traces: Iterable[Trace]) -> tuple[list[dict], ExportStats]:
        """Return (records, stats). Each record is `{"messages": [...]}`."""
        stats = ExportStats()
        seen_keys: set[str] = set()
        records: list[dict] = []
        for t in traces:
            stats.seen += 1
            try:
                if not (t.prompt and t.completion):
                    stats.dropped_short += 1
                    continue
                if not self._accept(t, stats):
                    continue
                key = self._prompt_key(t.system, t.prompt)
                if key in seen_keys:
                    stats.dropped_dup += 1
                    continue
                seen_keys.add(key)
                msgs = []
                if t.system:
                    msgs.append({"role": "system", "content": t.system})
                msgs.append({"role": "user", "content": t.prompt})
                msgs.append({"role": "assistant", "content": t.completion})
                rec = {"messages": msgs}
                if t.model:
                    rec["teacher"] = t.model
                records.append(rec)
                stats.kept += 1
            except Exception:  # noqa: BLE001 — one bad trace never sinks the export
                continue
        return records, stats

    def to_jsonl(self, traces: Iterable[Trace]) -> tuple[str, ExportStats]:
        records, stats = self.build(traces)
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        return (body + "\n" if body else ""), stats

    def write_jsonl(self, traces: Iterable[Trace], path: str) -> ExportStats:
        body, stats = self.to_jsonl(traces)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
        except Exception:  # noqa: BLE001
            stats.reasons.append("write failed — dataset not persisted")
        return stats


def enabled() -> bool:
    """`cfg.llm.distillation_export` — default OFF (an ops/eval action, not a
    hot-path feature)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "llm", None), "distillation_export", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = ["Trace", "ExportStats", "DistillationExporter", "enabled"]
