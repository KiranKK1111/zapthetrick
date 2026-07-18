"""Graded IT/CS task bank for the Claude-comparison eval (P2-2).

~50 prompts spanning the in-scope domains — bug-fix, build-from-spec, code-gen,
debugging, architecture, doc-gen — each carrying an OBJECTIVE rubric (`Gate`s)
so any model's free-form answer gets a comparable 0..1 score with no human in
the loop. The gates check verifiable surface properties (a fix uses a
parameterised query, an architecture answer names trade-offs, a doc has the
requested sections). They are intentionally lenient on phrasing and strict on
substance, so they reward correct *engineering* rather than wording.

`default_task_bank()` returns the whole set; `task_bank(categories=...,
difficulties=...)` filters it. Used by `model_eval.run_task_bank` to score one
or more models and by the `/api/eval/*` diagnostics endpoints.
"""
from __future__ import annotations

from app.eval.scoring import (
    GradedTask,
    contains_all,
    contains_any,
    contains_none,
    has_code_block,
    has_sections,
    min_words,
    regex_present,
)


def _bugfix() -> list[GradedTask]:
    return [
        GradedTask(
            id="bugfix/sql-injection",
            category="bugfix", difficulty="standard",
            prompt=(
                "This Python function is vulnerable. Fix it and show the "
                "corrected code:\n\n"
                "def get_user(db, name):\n"
                "    q = \"SELECT * FROM users WHERE name = '\" + name + \"'\"\n"
                "    return db.execute(q).fetchone()"
            ),
            gates=[
                has_code_block(weight=1.0),
                contains_any("?", "%s", ":name", "parameter", weight=2.0),
                contains_none("' + name + '", weight=1.5),
            ],
        ),
        GradedTask(
            id="bugfix/off-by-one",
            category="bugfix", difficulty="easy",
            prompt=(
                "Fix the off-by-one bug so it prints 1..n inclusive:\n\n"
                "for i in range(1, n):\n    print(i)"
            ),
            gates=[
                has_code_block(),
                regex_present(r"range\(\s*1\s*,\s*n\s*\+\s*1\s*\)", weight=2.0),
            ],
        ),
        GradedTask(
            id="bugfix/null-deref",
            category="bugfix", difficulty="standard",
            prompt=(
                "This JS crashes when `user` is null. Make it safe:\n\n"
                "function greet(user){ return 'Hi ' + user.name; }"
            ),
            gates=[
                has_code_block(),
                contains_any("?.", "user &&", "if (user", "!user", "== null",
                             "=== null", "optional chaining", weight=2.0),
            ],
        ),
        GradedTask(
            id="bugfix/race-condition",
            category="bugfix", difficulty="hard",
            prompt=(
                "Two threads increment a shared counter and the total is wrong. "
                "Explain the root cause and give a corrected Python snippet."
            ),
            gates=[
                contains_any("lock", "mutex", "atomic", "threading.Lock",
                             "synchron", weight=2.0),
                contains_any("race", "not atomic", "read-modify-write",
                             "data race", weight=1.5),
                has_code_block(),
            ],
        ),
        GradedTask(
            id="bugfix/memory-leak",
            category="bugfix", difficulty="hard",
            prompt=(
                "A Node.js service's memory grows unbounded over days. List the "
                "most likely causes and how you'd confirm each."
            ),
            gates=[
                contains_any("listener", "event", "closure", "global", "cache",
                             "timer", "setInterval", weight=2.0),
                contains_any("heap snapshot", "heap dump", "--inspect",
                             "profil", "clinic", weight=1.5),
                min_words(60),
            ],
        ),
    ]


def _build_from_spec() -> list[GradedTask]:
    return [
        GradedTask(
            id="build/rest-crud",
            category="build_from_spec", difficulty="standard",
            prompt=(
                "Write a minimal FastAPI app exposing CRUD for a `Note` "
                "(id, title, body) using an in-memory store. Include all four "
                "routes."
            ),
            gates=[
                has_code_block(weight=1.0),
                contains_all("FastAPI", weight=1.0),
                regex_present(r"@app\.(get|post)", weight=1.0),
                contains_all("post", "get", "put", "delete", weight=2.0),
            ],
        ),
        GradedTask(
            id="build/rate-limiter",
            category="build_from_spec", difficulty="hard",
            prompt=(
                "Implement a token-bucket rate limiter class in Python: "
                "configurable capacity and refill rate, a thread-safe "
                "`allow()` method. Show the code."
            ),
            gates=[
                has_code_block(),
                contains_all("class", weight=1.0),
                contains_any("capacity", "tokens", "refill", weight=1.5),
                contains_any("lock", "time", weight=1.0),
            ],
        ),
        GradedTask(
            id="build/binary-search",
            category="build_from_spec", difficulty="easy",
            prompt=(
                "Write an iterative binary search in Python that returns the "
                "index or -1. Include a one-line complexity note."
            ),
            gates=[
                has_code_block(),
                contains_any("mid", "lo", "hi", "low", "high", weight=1.0),
                contains_any("O(log", "logarithmic", weight=1.5),
            ],
        ),
        GradedTask(
            id="build/debounce",
            category="build_from_spec", difficulty="standard",
            prompt=(
                "Write a JavaScript `debounce(fn, wait)` higher-order function "
                "and explain when to use it."
            ),
            gates=[
                has_code_block(),
                contains_any("setTimeout", "clearTimeout", weight=2.0),
                min_words(30),
            ],
        ),
        GradedTask(
            id="build/lru-cache",
            category="build_from_spec", difficulty="hard",
            prompt=(
                "Implement an LRU cache with O(1) get/put in Python and explain "
                "the data structures used."
            ),
            gates=[
                has_code_block(),
                contains_any("OrderedDict", "doubly", "linked list",
                             "hash map", "dict", weight=2.0),
                contains_any("O(1)", "constant time", weight=1.0),
            ],
        ),
    ]


def _codegen() -> list[GradedTask]:
    return [
        GradedTask(
            id="codegen/sql-join",
            category="codegen", difficulty="standard",
            prompt=(
                "Write a SQL query: for each customer, their total order amount "
                "in 2024, only customers who spent over 1000, highest first."
            ),
            gates=[
                contains_all("select", "from", weight=1.0),
                contains_any("join", weight=1.0),
                contains_any("group by", weight=1.5),
                contains_any("having", "where", weight=1.0),
                contains_any("order by", weight=1.0),
            ],
        ),
        GradedTask(
            id="codegen/regex-email",
            category="codegen", difficulty="easy",
            prompt="Give a regex to validate a basic email and explain each part.",
            gates=[
                regex_present(r"@", weight=1.0),
                contains_any("\\.", "[a-z", "\\w", "+", weight=1.0),
                min_words(20),
            ],
        ),
        GradedTask(
            id="codegen/dockerfile",
            category="codegen", difficulty="standard",
            prompt=(
                "Write a production Dockerfile for a Python FastAPI app using a "
                "slim base image and a non-root user."
            ),
            gates=[
                contains_all("FROM", weight=1.0),
                contains_any("slim", "alpine", weight=1.0),
                contains_any("USER", "useradd", "adduser", weight=1.5),
                contains_any("CMD", "ENTRYPOINT", "uvicorn", weight=1.0),
            ],
        ),
        GradedTask(
            id="codegen/github-action",
            category="codegen", difficulty="standard",
            prompt=(
                "Write a GitHub Actions workflow that runs pytest on every push "
                "to main using Python 3.12."
            ),
            gates=[
                contains_all("on:", "jobs:", weight=1.5),
                contains_any("actions/checkout", weight=1.0),
                contains_any("pytest", weight=1.0),
                contains_any("3.12", weight=1.0),
            ],
        ),
        GradedTask(
            id="codegen/recursion",
            category="codegen", difficulty="easy",
            prompt="Write a recursive factorial in Python with a base case.",
            gates=[
                has_code_block(),
                regex_present(r"def\s+\w+", weight=1.0),
                contains_any("return 1", "n == 0", "n <= 1", "base case",
                             weight=1.5),
            ],
        ),
    ]


def _debugging() -> list[GradedTask]:
    return [
        GradedTask(
            id="debug/stacktrace",
            category="debugging", difficulty="standard",
            prompt=(
                "A Python app throws `KeyError: 'user_id'` when handling a "
                "webhook. Walk through how you'd diagnose and fix it."
            ),
            gates=[
                contains_any(".get(", "in payload", "KeyError", "default",
                             "validate", weight=2.0),
                min_words(40),
            ],
        ),
        GradedTask(
            id="debug/slow-query",
            category="debugging", difficulty="hard",
            prompt=(
                "A Postgres query that was fast is now slow after data growth. "
                "How do you find and fix the bottleneck?"
            ),
            gates=[
                contains_any("explain", "explain analyze", "query plan",
                             weight=2.0),
                contains_any("index", weight=1.5),
                contains_any("seq scan", "sequential scan", "vacuum",
                             "analyze", "statistics", weight=1.0),
            ],
        ),
        GradedTask(
            id="debug/cors",
            category="debugging", difficulty="standard",
            prompt=(
                "The browser reports a CORS error calling my API. Explain why "
                "and how to fix it on the server."
            ),
            gates=[
                contains_any("Access-Control-Allow-Origin", "CORS middleware",
                             "allow_origins", "preflight", weight=2.0),
                min_words(30),
            ],
        ),
        GradedTask(
            id="debug/flaky-test",
            category="debugging", difficulty="hard",
            prompt=(
                "A test passes locally but fails intermittently in CI. List the "
                "common causes and how to make it deterministic."
            ),
            gates=[
                contains_any("timing", "race", "order", "shared state",
                             "time", "random", "seed", "async", weight=2.0),
                min_words(50),
            ],
        ),
    ]


def _architecture() -> list[GradedTask]:
    return [
        GradedTask(
            id="arch/scale-reads",
            category="architecture", difficulty="hard",
            prompt=(
                "A read-heavy web app's database is the bottleneck. Propose an "
                "approach and name the trade-offs."
            ),
            gates=[
                contains_any("read replica", "cache", "redis", "cdn",
                             "denormal", "shard", weight=2.0),
                contains_any("trade-off", "tradeoff", "consistency",
                             "stale", "complexity", "downside", weight=2.0),
                min_words(60),
            ],
        ),
        GradedTask(
            id="arch/microservices",
            category="architecture", difficulty="hard",
            prompt=(
                "When should a team split a monolith into microservices, and "
                "when should they not? Give a balanced answer."
            ),
            gates=[
                contains_any("team", "scal", "deploy", "boundary", "domain",
                             weight=1.5),
                contains_any("not", "avoid", "premature", "overhead",
                             "complexity", "monolith", weight=2.0),
                min_words(70),
            ],
        ),
        GradedTask(
            id="arch/idempotency",
            category="architecture", difficulty="hard",
            prompt=(
                "Design idempotent payment processing so a retried request "
                "never double-charges. Describe the mechanism."
            ),
            gates=[
                contains_any("idempotency key", "idempotency-key",
                             "unique key", "dedup", weight=2.0),
                contains_any("store", "database", "lookup", "exists",
                             weight=1.0),
                min_words(50),
            ],
        ),
        GradedTask(
            id="arch/queue-vs-sync",
            category="architecture", difficulty="standard",
            prompt=(
                "When would you put a message queue between two services "
                "instead of a synchronous call? Trade-offs?"
            ),
            gates=[
                contains_any("async", "decoupl", "buffer", "spike", "retry",
                             "back-pressure", "backpressure", weight=2.0),
                contains_any("latency", "complexity", "eventual",
                             "trade-off", "tradeoff", weight=1.5),
                min_words(50),
            ],
        ),
        GradedTask(
            id="arch/cap-theorem",
            category="architecture", difficulty="expert",
            prompt=(
                "Explain the CAP theorem and give a concrete example of a "
                "CP system and an AP system."
            ),
            gates=[
                contains_all("consistency", "availability", weight=1.5),
                contains_any("partition", weight=1.5),
                contains_any("example", "e.g.", "such as", weight=1.0),
                min_words(60),
            ],
        ),
        GradedTask(
            id="arch/auth-design",
            category="architecture", difficulty="hard",
            prompt=(
                "Compare session cookies vs JWTs for a web app's auth. When is "
                "each the better choice?"
            ),
            gates=[
                contains_all("session", "jwt", weight=1.5),
                contains_any("revoke", "stateless", "expire", "refresh",
                             "storage", weight=2.0),
                min_words(60),
            ],
        ),
    ]


def _docgen() -> list[GradedTask]:
    return [
        GradedTask(
            id="docgen/readme",
            category="docgen", difficulty="standard",
            prompt=(
                "Write a README for a CLI tool `fastgrep` that searches files "
                "by regex. Include Installation, Usage, and Examples sections."
            ),
            gates=[
                has_sections("Installation", "Usage", weight=2.0),
                contains_any("example", weight=1.0),
                has_code_block(),
            ],
        ),
        GradedTask(
            id="docgen/api-doc",
            category="docgen", difficulty="standard",
            prompt=(
                "Document a `POST /users` endpoint: request body, response, and "
                "error codes."
            ),
            gates=[
                contains_any("request", "body", weight=1.0),
                contains_any("response", weight=1.0),
                contains_any("400", "404", "409", "error", "status",
                             weight=1.5),
            ],
        ),
        GradedTask(
            id="docgen/adr",
            category="docgen", difficulty="standard",
            prompt=(
                "Write an Architecture Decision Record for choosing Postgres "
                "over MongoDB. Use Context, Decision, Consequences."
            ),
            gates=[
                has_sections("Context", "Decision", "Consequences", weight=3.0),
                min_words(80),
            ],
        ),
        GradedTask(
            id="docgen/runbook",
            category="docgen", difficulty="standard",
            prompt=(
                "Write a runbook for responding to a 'service down' alert: "
                "detection, triage, mitigation, escalation."
            ),
            gates=[
                contains_all("triage", weight=1.0),
                contains_any("escalat", weight=1.0),
                contains_any("mitigat", "rollback", "restart", weight=1.5),
                min_words(60),
            ],
        ),
    ]


def _explain() -> list[GradedTask]:
    return [
        GradedTask(
            id="explain/async-await",
            category="explain", difficulty="standard",
            prompt=(
                "Explain async/await in Python to a developer who knows threads "
                "but not the event loop."
            ),
            gates=[
                contains_any("event loop", "coroutine", "await", "non-block",
                             weight=2.0),
                min_words(50),
            ],
        ),
        GradedTask(
            id="explain/big-o",
            category="explain", difficulty="easy",
            prompt="Explain the difference between O(n) and O(n^2) with an example.",
            gates=[
                contains_any("linear", "quadratic", "nested loop", "doubl",
                             weight=2.0),
                min_words(30),
            ],
        ),
        GradedTask(
            id="explain/git-rebase",
            category="explain", difficulty="standard",
            prompt="Explain the difference between git merge and git rebase.",
            gates=[
                contains_all("merge", "rebase", weight=1.5),
                contains_any("history", "linear", "commit", "replay",
                             weight=1.5),
                min_words(40),
            ],
        ),
    ]


def default_task_bank() -> list[GradedTask]:
    """The full graded benchmark across all in-scope IT/CS categories."""
    return [
        *_bugfix(),
        *_build_from_spec(),
        *_codegen(),
        *_debugging(),
        *_architecture(),
        *_docgen(),
        *_explain(),
    ]


def task_bank(*, categories: list[str] | None = None,
              difficulties: list[str] | None = None) -> list[GradedTask]:
    """The task bank, optionally filtered by category and/or difficulty."""
    tasks = default_task_bank()
    if categories:
        cats = {c.lower() for c in categories}
        tasks = [t for t in tasks if t.category.lower() in cats]
    if difficulties:
        diffs = {d.lower() for d in difficulties}
        tasks = [t for t in tasks if t.difficulty.lower() in diffs]
    return tasks


def categories() -> list[str]:
    return sorted({t.category for t in default_task_bank()})


__all__ = ["default_task_bank", "task_bank", "categories"]
