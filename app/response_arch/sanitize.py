"""Strip leaked chain-of-thought / reasoning from a model's visible text.

Primary defense lives in the provider adapters, which route reasoning
models' `reasoning`/`reasoning_content` deltas away from the answer
stream (see `app/llm/providers/openai_compat.py`). This module is the
belt-and-suspenders pass for the *other* way reasoning leaks: some
providers embed it inline in `content` using either:

  * `<think>…</think>` / `<thinking>…</thinking>` tags (DeepSeek-R1, Qwen), or
  * GPT-OSS "harmony" channel tokens
    (`<|channel|>analysis<|message|>…<|end|>` … `<|channel|>final<|message|>`).

`strip_reasoning` removes those, keeping only the final answer. It is
conservative: if stripping would empty the text, the original is kept.
"""
from __future__ import annotations

import re

# <think>…</think> and <thinking>…</thinking>, case-insensitive, multi-line.
_THINK_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# A dangling open tag with no close (truncated stream) → drop to end.
_THINK_OPEN = re.compile(
    r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)

# GPT-OSS harmony: keep only the text of the final channel if present.
_HARMONY_FINAL = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|(?:end|return|start)\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Any leftover harmony control tokens.
_HARMONY_TOKENS = re.compile(r"<\|[a-zA-Z_]+\|>")
# A non-final harmony channel block (analysis/commentary) → remove wholesale.
_HARMONY_NONFINAL = re.compile(
    r"<\|channel\|>\s*(?:analysis|commentary)\s*<\|message\|>.*?(?:<\|(?:end|start)\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def strip_reasoning(text: str) -> str:
    """Return `text` with leaked reasoning blocks removed."""
    if not text or ("<" not in text):
        return text
    out = text

    # 1. Harmony: if a final channel exists, that IS the answer.
    finals = _HARMONY_FINAL.findall(out)
    if finals:
        out = "\n".join(f.strip() for f in finals if f.strip()) or out
    else:
        out = _HARMONY_NONFINAL.sub("", out)
    out = _HARMONY_TOKENS.sub("", out)

    # 2. <think> blocks (paired, then any dangling open tag).
    out = _THINK_BLOCK.sub("", out)
    out = _THINK_OPEN.sub("", out)

    out = out.strip()
    # Never return empty — fall back to the original if we stripped too much.
    return out or text.strip()


__all__ = ["strip_reasoning"]
