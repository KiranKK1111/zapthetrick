"""Resume-aware hierarchical chunker.

Levels (Architecture.md §3):
  L1 — sections   (Experience, Skills, Projects, Education, ...)
  L2 — items      (one job, one project, one degree)
  L3 — bullets    (one bullet / one sentence)
  L4 — sliding window (500 tokens, 50 overlap)

Each chunk records its parent chain so retrieval can do "small-to-big":
match on a precise L3/L4 chunk, return the L2 parent block to the LLM.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable

# Section headings we recognize in a resume. Loose — we'll fall through
# to "Other" rather than break on a missing header.
_SECTION_RE = re.compile(
    r"^\s*(experience|work\s+experience|employment|projects|skills?|"
    r"education|certifications?|publications?|awards?|summary|"
    r"objective|profile)\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    """One unit on the hierarchical chunker output."""
    id: str
    level: int                       # 1..4
    text: str
    section_type: str = ""
    parent_id: str | None = None
    entity_tags: list[str] = field(default_factory=list)
    chunk_summary: str = ""          # populated by ingest.py if asked


class HierarchicalChunker:
    """Build the 4-level chunk tree for a single document.

    Stateless; call [chunk] with the raw text and get a flat list of
    [Chunk]s back. The tree is recoverable from `parent_id`.
    """

    def __init__(
        self,
        *,
        sliding_window_tokens: int = 500,
        sliding_overlap_tokens: int = 50,
    ) -> None:
        self.window = sliding_window_tokens
        self.overlap = sliding_overlap_tokens

    def chunk(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section_name, section_body in self._split_sections(text):
            section_id = self._new_id()
            chunks.append(
                Chunk(
                    id=section_id,
                    level=1,
                    text=section_body,
                    section_type=section_name,
                )
            )
            for item in self._split_items(section_body):
                item_id = self._new_id()
                chunks.append(
                    Chunk(
                        id=item_id,
                        level=2,
                        text=item,
                        section_type=section_name,
                        parent_id=section_id,
                    )
                )
                for bullet in self._split_bullets(item):
                    chunks.append(
                        Chunk(
                            id=self._new_id(),
                            level=3,
                            text=bullet,
                            section_type=section_name,
                            parent_id=item_id,
                        )
                    )
                for window in self._sliding_windows(item):
                    chunks.append(
                        Chunk(
                            id=self._new_id(),
                            level=4,
                            text=window,
                            section_type=section_name,
                            parent_id=item_id,
                        )
                    )
        return chunks

    # ---- helpers -----------------------------------------------------
    def _split_sections(self, text: str) -> Iterable[tuple[str, str]]:
        # Split by recognized section headings; everything before the first
        # heading falls into "Summary".
        positions = [(m.start(), m.group(1)) for m in _SECTION_RE.finditer(text)]
        if not positions:
            yield ("Other", text.strip())
            return
        first_start = positions[0][0]
        if first_start > 0:
            yield ("Summary", text[:first_start].strip())
        for i, (start, name) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[start:end]
            # Strip the heading itself off the body.
            body = _SECTION_RE.sub("", body, count=1).strip()
            yield (name.title(), body)

    def _split_items(self, section_body: str) -> list[str]:
        # Items are separated by blank lines.
        items = [b.strip() for b in re.split(r"\n\s*\n", section_body) if b.strip()]
        return items or [section_body]

    def _split_bullets(self, item: str) -> list[str]:
        # Lines starting with -, *, • or numbered.
        out: list[str] = []
        for line in item.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^[-*•]\s+", stripped):
                out.append(re.sub(r"^[-*•]\s+", "", stripped))
            elif re.match(r"^\d+\.\s+", stripped):
                out.append(re.sub(r"^\d+\.\s+", "", stripped))
        return out

    def _sliding_windows(self, item: str) -> list[str]:
        # TODO: token-aware via [tokenization.tokenizers]. For now,
        # approximate with whitespace tokens.
        words = item.split()
        if len(words) <= self.window:
            return []
        out: list[str] = []
        step = self.window - self.overlap
        for i in range(0, len(words) - self.overlap, step):
            out.append(" ".join(words[i : i + self.window]))
        return out

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:12]
