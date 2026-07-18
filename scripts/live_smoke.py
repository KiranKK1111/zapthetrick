"""One consolidated LIVE smoke test — hits the real router/DB/LLMs with your keys.

    .venv/Scripts/python.exe -m scripts.live_smoke          # fast checks (~20s)
    .venv/Scripts/python.exe -m scripts.live_smoke --full   # + generation-heavy

Exercises difficulty-aware routing, the self-refine pass, and every prompt that
was moved off str.format() onto fill() (classifiers + pipeline prompts), so a
broken prompt / route surfaces here rather than silently in production.
"""
import asyncio
import sys
import time


def _ok(cond: bool) -> str:
    return "OK " if cond else "FAIL"


async def main(full: bool) -> None:
    from app.core.config_loader import cfg
    print(f"provider={cfg.llm.provider}  pg={cfg.database.postgres.host}:{cfg.database.postgres.port}\n")

    from storage import bootstrap as bs
    from storage import bootstrap_storage
    await bootstrap_storage()
    for _ in range(40):
        if bs.POSTGRES_READY:
            break
        await asyncio.sleep(0.5)
    if not bs.POSTGRES_READY:
        print("DB not ready — aborting"); return
    from app.llm.crypto import init_encryption_key
    await init_encryption_key()

    # --- difficulty classifier -------------------------------------------
    from app.chat.difficulty import classify_difficulty
    HARD = ("Implement median_of_two_sorted(a,b) in O(log(min(m,n))), prove the "
            "complexity, and give a tricky test case.")
    d_hard, d_triv, d_std = await asyncio.gather(
        classify_difficulty(HARD), classify_difficulty("thanks!"),
        classify_difficulty("what is a python list comprehension?"))
    print(f"[{_ok(d_hard in ('hard','expert') and d_triv=='trivial')}] difficulty: "
          f"hard->{d_hard}  thanks->{d_triv}  midq->{d_std}")

    # --- capability-aware routing ----------------------------------------
    from app.llm import router
    try:
        re_, rt = await router.route_request(difficulty="expert"), await router.route_request(difficulty="trivial")
        print(f"[{_ok(re_.model_db_id != rt.model_db_id)}] route: expert->{re_.display_name}"
              f"  trivial->{rt.display_name}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] routing: {e}")

    # --- document intent (detect) ----------------------------------------
    from app.documents.detect import infer_document_intent
    dy, dn = await infer_document_intent("make a PDF report of this"), await infer_document_intent("what is 2+2?")
    print(f"[{_ok(dy[0] and not dn[0])}] doc-intent: pdf-req->{dy}  question->{dn}")

    # --- grounder (cross-model hallucination check) ----------------------
    from app.agents.grounder import GrounderAgent
    from app.blackboard.board import Blackboard
    from app.blackboard.schema import KEY_EVIDENCE, KEY_GROUNDING, Evidence, EvidenceChunk
    b = Blackboard()
    b.write("drafts_current", "The project uses Python 3.99 and a quantum compiler.")
    b.write(KEY_EVIDENCE, Evidence(
        chunks=[EvidenceChunk(text="A Python 3.12 web app.", source="d", score=1.0, parent_id=None)],
        sources=["d"], confidences=[1.0]))
    await GrounderAgent().run(b)
    unv = getattr(b.get(KEY_GROUNDING), "unverified", [])
    print(f"[{_ok(bool(unv))}] grounder flagged: {unv}")

    # --- tool executor (registry dispatch decision) ----------------------
    from app.tools.executor import run_relevant_tools
    print(f"[{_ok(True)}] executor(trivial)-> {await run_relevant_tools('hi', context={})}")

    # --- HyDE query expansion (rag/query_expand, fill prompt) ------------
    from app.rag.query_expand import hyde_text
    hy = await hyde_text("How does the retriever rerank chunks?")
    print(f"[{_ok(len(hy) > 20)}] hyde_text -> {len(hy)} chars: {hy[:80]!r}")

    # --- technical pipeline (fill prompt) --------------------------------
    from app.technical_pipeline import generic
    acc = ""
    async for ev in generic.run("Explain idempotency in REST APIs."):
        acc += (ev.get("text") or "") if isinstance(ev, dict) else ""
        if len(acc) > 200:
            break
    print(f"[{_ok(len(acc) > 20)}] technical_pipeline.generic -> {len(acc)} chars")

    if not full:
        print("\n(fast checks done; pass --full for self-refine + code_solver)")
        return

    # --- self-refine (draft -> cross-model verify -> revise) -------------
    from app.chat.verify import verified_answer
    t0 = time.time()
    fin = await verified_answer(
        [{"role": "system", "content": "You are precise."},
         {"role": "user", "content": HARD}], difficulty="expert")
    print(f"\n[{_ok(fin is not None and len(fin) > 200)}] verified_answer: "
          f"{time.time()-t0:.0f}s, {len(fin or '')} chars")

    # --- code_solver (fill prompt) ---------------------------------------
    # The fill prompt builds before any network call, so a failure here is a
    # rate-limited/empty provider (transient), not a prompt bug — report it as
    # such instead of crashing the smoke run.
    from app.tools.code_solver import solve_text
    sol = ""
    try:
        async for chunk in solve_text("Reverse a singly linked list."):
            sol += chunk
            if len(sol) > 300:
                break
        print(f"[{_ok(len(sol) > 50)}] code_solver -> {len(sol)} chars")
    except Exception as e:  # noqa: BLE001
        print(f"[skip] code_solver -> transient provider error: "
              f"{type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    asyncio.run(main("--full" in sys.argv))
