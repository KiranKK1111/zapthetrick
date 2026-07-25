"""Prompt assembly with stable-prefix caching (vNext §8.1).

One ``PromptAssembler`` owns prompt construction so every call site orders
content by VOLATILITY — most stable first — and a change in any layer only
invalidates the layers below it. That ordering is what lets providers serve a
cached prefill (Anthropic ``cache_control`` breakpoints; OpenRouter/DeepSeek/
Gemini implicit prefix caching) instead of re-prefilling an almost-identical
prefix every turn.

    ┌─ CACHED PREFIX (stable across the session) ────────────────────────┐
    │ L0 persona            + response contracts     changes: ~never      │
    │ L1 mode / band directives (§4.3)               changes: per session │
    │ L2 project instructions + standing prefs       changes: per project │
    │ L3 memory digest + candidate profile           changes: per session │
    │ L4 compacted history summary (§8.4)            changes: per compact │
    ├─ cache breakpoint ─────────────────────────────────────────────────┤
    │ L5 RAG context for THIS turn (§8.3)            changes: per turn    │
    │ L6 recent raw turns + current user message     changes: per turn    │
    └────────────────────────────────────────────────────────────────────┘

Hard requirement — **byte-stability of L0–L4**: no timestamps, no random ids, no
dict-ordering drift. Callers serialize any structured layer with sorted keys
before handing it in; this module only ever joins the given strings with a fixed
separator. ``prefix_hash()`` is therefore stable turn-to-turn whenever the inputs
are, which is exactly the signal the router's session-sticky rung (§2.4/T0) and
the answer cache (§3.6) key on.

Fail-safe: the assembler is a pure function of its inputs. A provider that
ignores caching sees identical messages, just without the prefill discount —
there is no correctness dependence on the cache.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

_SEP = "\n\n"

# The stable, cached-prefix layers, ordered most-stable-first (L0 → L4).
_STABLE = ("persona", "mode", "project", "memory", "history_summary")


@dataclass
class PromptAssembler:
    # ── stable prefix (L0–L4) ────────────────────────────────────────────
    persona: str = ""            # L0 — system persona + response contracts
    mode: str = ""               # L1 — band / mode directives
    project: str = ""            # L2 — project instructions + standing prefs
    memory: str = ""             # L3 — memory digest + candidate profile
    history_summary: str = ""    # L4 — compacted history summary
    # ── per-turn (below the breakpoint) ──────────────────────────────────
    rag: str = ""                # L5 — retrieved context for THIS turn
    recent: list[dict] = field(default_factory=list)  # L6 — recent raw turns
    user: str = ""               # L6 — the current user message

    # ── stable-prefix accounting ─────────────────────────────────────────
    def _stable_parts(self) -> list[tuple[str, str]]:
        """Non-empty stable layers in fixed L0→L4 order."""
        return [(n, getattr(self, n)) for n in _STABLE if getattr(self, n)]

    def stable_prefix(self) -> str:
        """The byte-stable cached prefix (L0–L4), deterministically joined."""
        return _SEP.join(text for _, text in self._stable_parts())

    def prefix_hash(self) -> str:
        """Content hash of the stable prefix — model-agnostic (the router keys
        the per-model cache by (model, prefix_hash))."""
        return hashlib.sha256(self.stable_prefix().encode("utf-8")).hexdigest()

    def layer_hashes(self) -> list[tuple[str, str]]:
        """Cumulative hash after each non-empty stable layer. Encodes the
        "a change in a layer only invalidates the layers below it" invariant:
        editing L2 leaves the L0 and L1 cumulative hashes untouched and changes
        L2, L3, L4 — so a provider (or our accounting) can reuse the longest
        matching prefix."""
        out: list[tuple[str, str]] = []
        acc = ""
        for name, text in self._stable_parts():
            acc = (acc + _SEP + text) if acc else text
            out.append((name, hashlib.sha256(acc.encode("utf-8")).hexdigest()))
        return out

    # ── message construction ─────────────────────────────────────────────
    def system_content(self) -> str:
        """The system message: stable prefix + this-turn RAG (L5), joined
        deterministically. RAG sits AFTER the prefix so the cached span is
        exactly L0–L4."""
        parts = []
        prefix = self.stable_prefix()
        if prefix:
            parts.append(prefix)
        if self.rag:
            parts.append(self.rag)
        return _SEP.join(parts)

    def build(self) -> list[dict]:
        """Assemble the provider message list: [system?] + recent turns + user.

        Recent turns are copied with the same ``role and content`` truthiness
        filter call sites already apply, so this is a drop-in for hand-built
        ``[{system}, *history, {user}]`` conversations."""
        msgs: list[dict] = []
        sc = self.system_content()
        if sc:
            msgs.append({"role": "system", "content": sc})
        for m in self.recent or []:
            r, c = m.get("role"), m.get("content")
            if r and c:
                msgs.append({"role": r, "content": c})
        if self.user:
            msgs.append({"role": "user", "content": self.user})
        return msgs

    def build_cached(self) -> tuple[list[dict], dict]:
        """Messages plus cache metadata the router/adapters consume:
        ``{prefix_hash, breakpoint_index, breakpoint_char}``. ``breakpoint_index``
        is the message that ends the cached span (the leading system message, or
        -1 if there is none); ``breakpoint_char`` is where L4 ends inside it (so
        an Anthropic adapter can split a cached block from the per-turn RAG)."""
        msgs = self.build()
        has_system = bool(msgs) and msgs[0].get("role") == "system"
        return msgs, {
            "prefix_hash": self.prefix_hash(),
            "breakpoint_index": 0 if has_system else -1,
            "breakpoint_char": len(self.stable_prefix()),
        }

    def anthropic_system_blocks(self) -> list[dict]:
        """The system message as Anthropic content blocks, marking the stable
        prefix (L0–L4) cacheable and leaving per-turn RAG ephemeral. Adapters
        that support ``cache_control`` use this; everyone else uses ``build()``.
        Empty prefix → a single plain block (no marker)."""
        blocks: list[dict] = []
        prefix = self.stable_prefix()
        if prefix:
            blocks.append({"type": "text", "text": prefix,
                           "cache_control": {"type": "ephemeral"}})
        if self.rag:
            blocks.append({"type": "text", "text": self.rag})
        return blocks


__all__ = ["PromptAssembler"]
