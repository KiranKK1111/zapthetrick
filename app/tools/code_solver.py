"""
Code-solver tool: structured coding-interview answers.

Given a problem statement (text and/or screenshot bytes), the solver
asks the configured `code_model` for a structured response with the
canonical sections — restatement, assumptions, approach, code,
walkthrough, complexity, edges, alternatives.

Two entry points:
  - `solve_text(problem)`  for typed problems
  - `solve_image(image_bytes)` for screenshots (uses cfg.llm.vision_model)

Both yield text chunks (so the route can SSE-stream tokens).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import AsyncGenerator

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.tools.registry import Tool, register
from app.core.prompt import fill

_SOLVER_SYSTEM_PROMPT = """You are a senior coding-interview assistant. The user has shown you a coding problem (either typed or as a screenshot of a problem page such as LeetCode). Read the problem carefully, then produce a *rigorous* structured response.

Hard requirements before you answer:
- Re-read the problem text. Pay attention to constraints (size limits, value ranges, sortedness, special characters).
- Decide what `{language}` means here: if the user supplied an explicit language, use it; if a code editor / function signature is visible in the image, use that exact language and match the given signature; otherwise default to Python.
- Do NOT invent constraints that are not present. Do NOT skip edge cases.

Use these exact section headings, in this order, each on its own line:

1. PROBLEM RESTATEMENT
   - One paragraph in your own words. Cite the input/output types and the constraints verbatim.

2. CLARIFYING ASSUMPTIONS
   - Bullet list. Cover anything the problem leaves ambiguous (empty inputs, negatives, duplicates, capacity, locale, ...).

3. APPROACH
   - Start with a brute-force solution and state its complexity.
   - Then derive the optimal approach step by step. Explain *why* it works (invariant / proof sketch). Mention the key data structure or trick that unlocks it.

4. CODE in {language}
   - Self-contained, compilable. Match the given signature if one was shown.
   - Variable names should be descriptive. Comments only where the logic is non-obvious.

5. WALKTHROUGH
   - Pick one of the provided sample inputs (or invent a small one).
   - Show the state of the key data structures at each non-trivial step.

6. COMPLEXITY
   - Time:  derive O(...) by counting work per iteration × number of iterations. State `n`, `k`, etc. explicitly. Justify in one line.
   - Space: derive O(...) by accounting for all auxiliary structures (call stack, hash maps, output). Justify in one line.
   - If recursion is used, include stack depth in the space analysis.

7. EDGE CASES
   - At least 4 distinct edge cases, with the expected behaviour for each. Include empty input, single element, all-equal, max-size, negatives/overflow where relevant.

{alternatives_section}
Output formatting:
- All code goes inside a fenced ```{language} block.
- Section headings on their own lines (the numbers and uppercase labels above).
- No filler like "Sure, here is..." or "Hope this helps".
- Keep the whole response under 1200 words, but never sacrifice correctness for length.
"""

_ALTERNATIVES_SECTION = (
    "8. ALTERNATIVE APPROACHES\n"
    "   - Up to two alternatives if they exist, each with code-free description "
    "and its full complexity. Skip this section if no realistic alternatives.\n\n"
)

def _build_system_prompt(language: str) -> str:
    return fill(_SOLVER_SYSTEM_PROMPT, 
        language=language,
        alternatives_section=_ALTERNATIVES_SECTION
        if cfg.code_solver.include_alternatives
        else "",
    )

async def solve_text(
    problem: str,
    language: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a structured solution to a typed coding problem.

    Capability-aware: the problem is classified, so a hard/expert one escalates
    to the strongest available model and runs the iterative verify→revise loop
    (deep-think across models) instead of always pinning `code_model`."""
    lang = language or cfg.code_solver.default_language
    messages = [
        {"role": "system", "content": _build_system_prompt(lang)},
        {"role": "user", "content": problem},
    ]
    model = cfg.llm.code_model or cfg.llm.model

    difficulty = "standard"
    try:
        if cfg.advanced_rag.difficulty_aware_routing:
            from app.chat.difficulty import classify_difficulty
            difficulty = await classify_difficulty(problem)
    except Exception:  # noqa: BLE001
        difficulty = "standard"

    # Hard/expert: draft → verify → revise across models, then stream the result.
    try:
        from app.chat.verify import chunk_text, verified_answer
        verified = await verified_answer(messages, difficulty=difficulty)
    except Exception:  # noqa: BLE001
        verified = None
    if verified is not None:
        for piece in chunk_text(verified):
            yield piece
        return

    # Otherwise stream directly; difficulty still biases the router toward a more
    # capable model for a hard turn without the extra loop cost.
    async for chunk in llm.stream_chat(
        messages, model=model, options={"difficulty": difficulty}
    ):
        yield chunk

# ---- Two-step prompts -------------------------------------------------
# Step 1 — OCR-only system prompt. The vision model is told to ONLY
# transcribe what it sees, in a structured layout. Reasoning is
# forbidden here so a small vision model doesn't fall over trying to
# solve the problem from a noisy image.
#
# CRITICAL: do NOT put any concrete function names, parameter names, or
# problem-statement phrasing as "examples" in this prompt. Small vision
# models (e.g. llava:7b) will parrot them back verbatim instead of
# reading what's actually in the screenshot. Keep all placeholders
# abstract.
_OCR_SYSTEM_PROMPT = """You are an OCR transcription assistant. The image is a screenshot of a coding-interview problem page (LeetCode, HackerRank, Codility, or similar). Read it carefully and output what is ACTUALLY visible — never invent text.

Output this EXACT structure with these EXACT headings, even if a section is empty. Use literal "NONE" for an empty section. Never fabricate content for a section.

=== TITLE ===
<the problem's title as it appears at the top of the page, or NONE>

=== FUNCTION SIGNATURE ===
<the signature line copied character-for-character from the code editor on screen, including return type, function name, and every parameter type+name. If no signature is visible, write NONE. Do NOT guess or invent a signature.>

=== PROBLEM STATEMENT ===
<the full problem description, copied verbatim. Do not paraphrase or shorten.>

=== EXAMPLES ===
<every input/output example shown, copied verbatim with original formatting>

=== CONSTRAINTS ===
<every constraint shown, copied verbatim (size bounds, value ranges, special characters)>

=== STARTER CODE ===
<any starter code block visible in the editor, copied verbatim with indentation>

Rules — violating any of these is a bug:
- Output ONLY what is visibly present in the image. If you cannot read a section, write NONE.
- Do NOT summarize. Do NOT solve. Do NOT add commentary, headers, or markdown beyond the five "=== HEADING ===" lines.
- The signature must match the screenshot character-for-character — downstream code generation depends on it.
- If the entire image is unreadable or contains no coding problem, output just: NONE
"""

_OCR_USER_PROMPT = (
    "Transcribe what you actually see in this screenshot using the structure "
    "in the system message. Read every section of the page (title, description, "
    "examples, constraints, code editor). Do not invent any signature or problem "
    "text — copy verbatim or write NONE."
)

# Step 2 — Reasoning system prompt. Takes the OCR output and produces the
# final solution. Mirrors the structure of WrongPathAI's
# CODING_TEXT_SYSTEM_PROMPT so the output is consistent with the source
# project's UX. Carefully avoids the words "OCR" / "transcribe" /
# "extract" — local reasoning models latch onto them and produce a long
# chain-of-thought about being an OCR assistant instead of solving.
_CODING_TEXT_SYSTEM_PROMPT = """Elite coding-interview copilot. The input text is a structured coding problem with === TITLE ===, === FUNCTION SIGNATURE ===, === PROBLEM STATEMENT ===, === EXAMPLES ===, === CONSTRAINTS ===, === STARTER CODE === sections.

CRITICAL — read the FUNCTION SIGNATURE section first. Your code's function name, return type, and parameter list MUST match it character-for-character. Do not invent a different signature. Do not change the function name. If no FUNCTION SIGNATURE is provided (or it says NONE), infer one from the STARTER CODE; if there's no starter either, pick a sensible signature from the problem statement.

Produce a clean, beautifully-formatted answer (GitHub Markdown, like a top-tier chat assistant) with EXACTLY these three sections, in this order:

## 1. Problem
Explain the problem clearly in your own words (2-4 sentences): what's being asked, the input and output, and the key constraints (size bounds, value ranges) that decide which algorithm is acceptable. Don't just copy the statement — make it crisp and easy to understand.

## 2. Solution
The OPTIMAL solution — best time AND space complexity achievable, scalable to the stated constraints. Put the complete, runnable code in a fenced block whose tag matches the signature's language:

```<language>
<the function with the EXACT signature from the FUNCTION SIGNATURE section — keep the class wrapper (class Solution { … } / class Result { … }) only if the signature shows one; descriptive names; comments only where logic is non-obvious>
```

Immediately after the code, give:
- **Time complexity:** O(...) — spell out what n / k / m mean and justify in one line.
- **Space complexity:** O(...) — account for ALL auxiliary memory (hash maps, recursion stack, output); justify in one line.
- **Optimizations & scaling:** 1-3 bullets — the memory/runtime optimizations you applied (in-place, early exit, avoiding extra allocations, streaming) and how it scales at the constraint limits. If a brute force is much simpler, name it and its complexity in one line so the trade-off is clear.

## 3. Explanation
Explain HOW the solution works so the candidate can defend it: the core idea / key insight, the data structure or technique that unlocks the optimal complexity, a short step-by-step of the logic, and a one-line trace over one of the EXAMPLES. Use short paragraphs and bullets — clear and skimmable.

Language detection: read the signature you were given.
  - `public ... foo(...)` -> java
  - `def foo(...):` -> python
  - `function foo(...)` or `const foo = (...) =>` -> javascript
  - `func foo(...)` -> go
  - `ListNode*` / `vector<...>` in the signature -> cpp

Hard rules — violating any of these is a bug:
- Use the three `##` section headers above (Problem / Solution / Explanation), each on its own line with a blank line around it.
- The opening ```<language> MUST start on its own line, with a blank line before it; the closing ``` on its own line.
- The function name and signature in your code MUST match the FUNCTION SIGNATURE section verbatim.
- Solve THE problem described in PROBLEM STATEMENT. Do not solve a similar-sounding problem from memory.
- No filler ("Sure, here's…", "Hope this helps"). Lead with the content.
- If the input says NONE everywhere or does not contain a coding problem, reply with one line saying so, then stop.
"""

@dataclass
class SolveStatus:
    """Status event yielded between phases of the two-step solve."""
    text: str

@dataclass
class SolveExtracted:
    """The OCR-extracted problem statement.

    Yielded once, right after the vision model returns, so the route
    layer can persist the description into the `solve_sessions` row.
    The Solve screen can use this same payload to display the captured
    problem to the user before the answer streams.
    """
    text: str

async def solve_image(
    image_bytes: bytes,
    language: str | None = None,
    extra_context: str | None = None,
    *,
    vision_model: str | None = None,
    code_model: str | None = None,
) -> AsyncGenerator:
    """Stream a structured solution from a screenshot.

    Args:
        image_bytes: raw screenshot bytes (PNG / JPEG).
        language: target language hint for the single-step path.
        extra_context: free-form note appended to the OCR / vision prompt.
        vision_model: per-call override for the OCR / single-step model.
            Falls back to cfg.llm.vision_model, then cfg.llm.model.
        code_model: per-call override for the reasoning model (two-step only).
            Falls back to cfg.llm.code_model, then cfg.llm.model.

    Dispatches to the two-step (OCR -> reason) pipeline when
    `cfg.code_solver.two_step_solve` is true (default), otherwise falls
    back to a single-step vision call.

    Yields a mix of `str` (token text for the UI) and `SolveStatus`
    (status updates the route can surface as SSE `status` events).
    """
    if cfg.code_solver.two_step_solve:
        async for item in _solve_image_two_step(
            image_bytes,
            language,
            extra_context,
            vision_model_override=vision_model,
            code_model_override=code_model,
        ):
            yield item
        return

    async for item in _solve_image_single_step(
        image_bytes,
        language,
        extra_context,
        vision_model_override=vision_model,
    ):
        yield item

# ---- Two-step pipeline -------------------------------------------------
async def _solve_image_two_step(
    image_bytes: bytes,
    language: str | None,
    extra_context: str | None,
    *,
    vision_model_override: str | None = None,
    code_model_override: str | None = None,
) -> AsyncGenerator:
    """Vision OCR -> structured text -> text/code reasoning model.

    Far more reliable on small local vision models (LLaVA, moondream)
    than asking them to read AND reason at once.
    """
    vision_model = vision_model_override or cfg.llm.vision_model or cfg.llm.model
    code_model = code_model_override or cfg.llm.code_model or cfg.llm.model

    # Step 1 — OCR. Silent; we don't stream this to the UI.
    yield SolveStatus(f"Reading problem with {vision_model}…")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    ocr_messages = [
        {"role": "system", "content": _OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _OCR_USER_PROMPT + (f"\n\nHint: {extra_context}" if extra_context else ""),
            "images": [encoded],
        },
    ]
    extracted = await llm.complete(
        ocr_messages,
        model=vision_model,
        options={
            "temperature": cfg.temperature.classifier,
            "num_predict": cfg.code_solver.ocr_max_tokens,
        },
    )
    cleaned = extracted.strip()

    # Validate that the OCR actually extracted *content*, not just the
    # section skeleton with NONE / empty bodies. Without this the
    # reasoning model sees only `=== HEADINGS ===` plus "NONE" and starts
    # solving the structure as if it were the problem (e.g. "write a
    # function that reads page['title'], page['description'], ..." —
    # treating my section labels as field names in a dict).
    real_content, content_chars = _ocr_real_content(cleaned)
    if (
        not cleaned
        or len(cleaned) < 20
        or cleaned.upper().strip() == "NONE"
        or content_chars < 50
    ):
        raise LLMError(
            f'Vision model "{vision_model}" could not read the screenshot '
            f"(extracted only {content_chars} chars of real content). "
            f"Pull a stronger vision model: `ollama pull qwen2.5vl:7b` "
            f"(best OCR) or `ollama pull moondream:latest`, then pick it "
            f"as the Vision model in Settings. Also confirm the problem "
            f"text is fully visible on screen — if our overlay covers the "
            f"problem area, move it or rebuild after the WDA_EXCLUDEFROM"
            f"CAPTURE change took effect."
        )

    # Surface the extracted problem so the route layer can persist it
    # (`SolveSession.description`) and the UI can render the captured
    # problem alongside the streamed solution. Routes consume this
    # event by isinstance-check; the Flutter side ignores it for now.
    yield SolveExtracted(cleaned)

    # Step 2 — Reasoning. Stream tokens.
    yield SolveStatus(f"Solving with {code_model}…")

    # The user message must make it absolutely clear that the "===" lines
    # are labels, not the problem. Otherwise generalist reasoning models
    # (llama3.1) read the section headings as field names and write code
    # that processes a JSON object with those fields — instead of solving
    # the coding problem inside the sections.
    reason_user = (
        "Below is the structured content of a coding-interview problem.\n"
        "\n"
        "FORMAT NOTE: lines starting and ending with three equals signs (e.g. "
        "`=== PROBLEM STATEMENT ===`) are SECTION LABELS that tell you what "
        "each block contains. They are NOT part of the problem. Do not treat "
        "them as field names, dict keys, function parameters, or anything to "
        "be processed. The PROBLEM you must solve is the prose between the "
        "labels, in particular the text under PROBLEM STATEMENT, EXAMPLES, "
        "and CONSTRAINTS, using the signature under FUNCTION SIGNATURE.\n"
        "\n"
        "Go straight to the three-section layout (## 1. Problem, ## 2. Solution "
        "with code + complexity + optimizations, ## 3. Explanation). Match the "
        "language of the function signature.\n"
        "\n"
        "------ BEGIN PROBLEM ------\n"
        f"{cleaned}\n"
        "------ END PROBLEM ------"
    )
    reason_messages = [
        {"role": "system", "content": _CODING_TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": reason_user},
    ]
    # Capability-aware: a hard/expert extracted problem escalates to the
    # strongest model and runs the deep-think loop, like the text path.
    difficulty = "standard"
    try:
        if cfg.advanced_rag.difficulty_aware_routing:
            from app.chat.difficulty import classify_difficulty
            difficulty = await classify_difficulty(cleaned)
    except Exception:  # noqa: BLE001
        difficulty = "standard"
    try:
        from app.chat.verify import chunk_text, verified_answer
        verified = await verified_answer(reason_messages, difficulty=difficulty)
    except Exception:  # noqa: BLE001
        verified = None
    if verified is not None:
        for piece in chunk_text(verified):
            yield piece
        return
    async for chunk in llm.stream_chat(
        reason_messages, model=code_model, options={"difficulty": difficulty}
    ):
        yield chunk

# ---- OCR-content validity helper --------------------------------------
def _ocr_real_content(ocr_text: str) -> tuple[bool, int]:
    """Strip the section skeleton out of OCR output and measure what remains.

    A well-formed OCR result has `=== HEADING ===` lines (skeleton) plus
    real text underneath. A failed OCR returns the skeleton with `NONE`
    or whitespace under every heading. Without filtering, len(ocr_text)
    is large but the *content* is empty — and the reasoning model
    treats the headings themselves as the task.

    Returns (has_real_content, real_content_char_count).
    """
    if not ocr_text:
        return False, 0
    real_lines: list[str] = []
    for line in ocr_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip "=== HEADING ===" markers entirely.
        if stripped.startswith("===") and stripped.endswith("==="):
            continue
        # Skip explicit NONE placeholders.
        if stripped.upper() == "NONE":
            continue
        real_lines.append(stripped)
    content = " ".join(real_lines)
    return len(content) >= 50, len(content)

# ---- Single-step fallback ---------------------------------------------
async def _solve_image_single_step(
    image_bytes: bytes,
    language: str | None,
    extra_context: str | None,
    *,
    vision_model_override: str | None = None,
) -> AsyncGenerator:
    """Original single-step solver: vision model reads + reasons in one call."""
    lang = language or cfg.code_solver.default_language
    user_text = (
        "The image is a screenshot of a coding-interview problem page "
        "(typically LeetCode). First, read the FULL problem text: title, "
        "description, every example, the constraints section, and any code "
        "editor / function signature that is visible. If a function signature "
        "is visible, your code MUST match it exactly (same name, same return "
        "type, same parameter names). If the editor shows a language, use "
        "that language; otherwise use the default. Do not skip the constraints "
        "section — it determines which algorithm is acceptable.\n\n"
        f"{extra_context or ''}"
    ).strip()
    encoded = base64.b64encode(image_bytes).decode("ascii")

    messages = [
        {"role": "system", "content": _build_system_prompt(lang)},
        {
            "role": "user",
            "content": user_text,
            "images": [encoded],
        },
    ]
    model = (
        vision_model_override
        or cfg.llm.vision_model
        or cfg.llm.code_model
        or cfg.llm.model
    )
    yield SolveStatus(f"Solving with {model}…")
    async for chunk in llm.stream_chat(messages, model=model):
        yield chunk

# ---- Tool registration (text-only; vision flows directly through routes) ----
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string", "description": "The coding problem statement."},
        "language": {
            "type": "string",
            "description": "Target language (default: python).",
        },
    },
    "required": ["problem"],
}

async def _tool_handler(*, problem: str, language: str | None = None) -> str:
    """Non-streaming variant for orchestrator tool-use (collects to string)."""
    out: list[str] = []
    async for chunk in solve_text(problem, language=language):
        out.append(chunk)
    return "".join(out).strip()

register(
    Tool(
        name="code_solver",
        description=(
            "Produce a structured solution to a coding/algorithms problem: "
            "restatement, approach, code, walkthrough, complexity, edges."
        ),
        input_schema=INPUT_SCHEMA,
        handler=_tool_handler,
    )
)
