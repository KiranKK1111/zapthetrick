"""Semantic-intent benchmark (user ask 2026-07-08: "rigorously check where
else the intent classification is going wrong").

Runs the REAL bge-m3 embedder over a labeled prompt corpus and reports every
misroute of `detect_intent_smart` (the production decision: semantic verdict
when sim ≥ threshold, else regex). The CI behavior corpus can't catch these —
it runs on the regex fallback because the embedder isn't loaded in tests —
so this harness is the live-path safety net. Run manually:

    python -m app.eval.semantic_intent_bench

Expected labels use the pre-gate's intent names. `accept` lists alternative
intents that are ALSO fine for genuinely dual-natured prompts (e.g. "how do I
sort a list in place" is knowledge-or-code_gen — either answers correctly and
neither triggers a wrong clarification).
"""
from __future__ import annotations

# (prompt, expected, {also acceptable})
CASES: list[tuple[str, str, set[str]]] = [
    # ---- code generation (the family that must ask for a language) --------
    ("give me a program for finding nth max number from an array",
     "code_generation", set()),
    ("write a login api", "code_generation", set()),
    # project_build is behaviorally equivalent here: both intents require the
    # missing stack, so either label produces the same (correct) clarify.
    ("write a rest api", "code_generation", {"project_build"}),
    ("create an endpoint for user signup", "code_generation", set()),
    ("implement a rate limiter", "code_generation", set()),
    ("write a function to merge two sorted lists", "code_generation", set()),
    ("give me the code for a linked list", "code_generation", set()),
    ("write a script to rename all files in a folder", "code_generation",
     set()),
    ("build a crud service for products", "code_generation",
     {"project_build"}),
    ("write a query to find duplicate emails", "code_generation", set()),
    # ---- knowledge ----------------------------------------------------------
    ("what is a hash map", "knowledge", set()),
    ("how does binary search work", "knowledge", set()),
    ("explain this code", "knowledge", set()),
    ("what's the difference between tcp and udp", "knowledge", {"comparison"}),
    ("why does this work the way it does", "knowledge", set()),
    ("what is dependency injection", "knowledge", set()),
    # ---- comparison ---------------------------------------------------------
    ("compare kafka and rabbitmq", "comparison", set()),
    ("react vs vue for a dashboard", "comparison", set()),
    ("pros and cons of graphql over rest", "comparison", set()),
    # ---- debugging ----------------------------------------------------------
    ("why is this throwing a null pointer exception", "debugging", set()),
    ("my function crashes on empty input", "debugging", set()),
    ("this test keeps failing, what's wrong", "debugging", set()),
    ("fix this stack trace", "debugging", set()),
    # ---- test generation ----------------------------------------------------
    ("write unit tests for this service", "test_generation", set()),
    ("add pytest coverage for this module", "test_generation", set()),
    ("generate test cases for the login flow", "test_generation", set()),
    # ---- docs ----------------------------------------------------------------
    ("write documentation for this module", "documentation", set()),
    ("generate a readme for the project", "documentation", set()),
    ("turn this into a word document", "documentation", set()),
    ("give me a pdf of the design", "documentation", set()),
    ("document this api", "documentation", set()),
    # ---- design ---------------------------------------------------------------
    ("design the architecture for a chat app", "design", set()),
    ("propose a database schema for orders", "design", set()),
    ("how should i model this data", "design", set()),
    # ---- project build ---------------------------------------------------------
    ("build me a todo app", "project_build", set()),
    ("create a full stack ecommerce site", "project_build", set()),
    ("scaffold a new web application", "project_build", set()),
    # ---- archive ---------------------------------------------------------------
    ("zip up the whole project", "archive", set()),
    ("compress this into a single file", "archive", set()),
    ("give me the archive of everything", "archive", set()),
    # ---- chitchat --------------------------------------------------------------
    ("hey there", "chitchat", set()),
    ("thanks, that was helpful", "chitchat", set()),
    ("good morning!", "chitchat", set()),
]


def run(embed_fn=None) -> dict:
    from app.clarify.intent_pipeline import detect_intent_smart

    failures: list[dict] = []
    for prompt, expected, accept in CASES:
        got = detect_intent_smart(prompt, embed_fn=embed_fn) \
            if embed_fn is not None else detect_intent_smart(prompt)
        ok = got == expected or got in accept
        if not ok:
            failures.append({"prompt": prompt, "expected": expected,
                             "got": got})
    return {
        "total": len(CASES),
        "correct": len(CASES) - len(failures),
        "accuracy": round((len(CASES) - len(failures)) / len(CASES), 4),
        "failures": failures,
    }


if __name__ == "__main__":
    import time

    from app.rag import embedder as emb
    emb.ensure_loading_in_background()
    for _ in range(90):
        if emb.is_ready():
            break
        time.sleep(2)
    print("embedder ready:", emb.is_ready())
    report = run()
    print({k: v for k, v in report.items() if k != "failures"})
    for f in report["failures"]:
        print("MISROUTE:", f)


__all__ = ["CASES", "run"]
