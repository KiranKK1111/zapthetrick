"""Pre-generated live answers (latency batch 2026-07-11, item #2).

When a resume is uploaded, the ~30 most common interview questions are
answered ONCE in the background — grounded in the profile, in the spoken
dictate-ready voice — and stored per resume with their question embeddings.
During the interview, an incoming profile question is matched by cosine
similarity and the prepared answer streams INSTANTLY: zero model latency for
exactly the questions that matter most, live generation for the long tail.

Storage: `data/prepared/{resume_id}.json` (no migration; dies with the data
dir). Everything is fail-open — a missing store / cold embedder / any error
means the normal generation path runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time

log = logging.getLogger("zapthetrick.live.prepared")

# The canonical bank: id → question. Broad coverage of the openers, HR
# staples, behavioral standards, and wrap-ups interviewers reuse verbatim.
QUESTION_BANK: dict[str, str] = {
    "about_yourself": "Tell me about yourself.",
    "walk_resume": "Walk me through your resume.",
    "background": "Tell me about your background.",
    "current_role": "Describe your current role and responsibilities.",
    "recent_project": "Tell me about a recent project you worked on.",
    "proud_project": "Which project are you most proud of and why?",
    "strengths": "What are your greatest strengths?",
    "weaknesses": "What are your weaknesses?",
    "strengths_weaknesses": "What are your strengths and weaknesses?",
    "why_hire": "Why should we hire you?",
    "why_company": "Why do you want to work for this company?",
    "why_role": "Why are you interested in this role?",
    "why_leaving": "Why are you leaving your current job?",
    "career_goals": "Where do you see yourself in five years?",
    "biggest_challenge": "Tell me about the biggest challenge you have "
                         "faced and how you handled it.",
    "conflict": "Tell me about a time you had a conflict with a teammate "
                "and how you resolved it.",
    "failure": "Tell me about a time you failed and what you learned.",
    "leadership": "Tell me about a time you showed leadership.",
    "pressure": "How do you handle pressure and tight deadlines?",
    "prioritize": "How do you prioritize your work when everything is "
                  "urgent?",
    "disagree_manager": "Tell me about a time you disagreed with your "
                        "manager.",
    "difficult_decision": "Describe a difficult decision you had to make "
                          "at work.",
    "team_or_alone": "Do you prefer working in a team or independently?",
    "stay_current": "How do you keep your skills up to date?",
    "tech_stack": "What technologies do you work with day to day?",
    "achievement": "What is your biggest professional achievement?",
    "mistake": "Tell me about a mistake you made and how you fixed it",
    "feedback": "How do you handle critical feedback?",
    "motivation": "What motivates you at work?",
    "salary": "What are your salary expectations?",
    "notice_period": "When can you start? What is your notice period?",
    "questions_for_us": "Do you have any questions for us?",
    # Extended coverage (2026-07-11 follow-up): more staples that ground
    # cleanly in a resume.
    "not_on_resume": "Tell me something about you that is not on your "
                     "resume.",
    "unique": "What makes you unique compared to other candidates?",
    "three_words": "How would you describe yourself in three words?",
    "manager_says": "What would your current manager say about you?",
    "coworkers_say": "How would your coworkers describe you?",
    "ideal_environment": "What is your ideal work environment?",
    "remote_or_office": "Do you prefer working remotely or in an office?",
    "career_choice": "Why did you choose this career path?",
    "education": "Tell me about your education and how it prepared you "
                 "for this role.",
    "resume_gap": "Can you explain the gap in your resume?",
    "least_liked": "What did you like least about your last job?",
    "above_beyond": "Tell me about a time you went above and beyond.",
    "technical_challenge": "Walk me through a difficult technical "
                           "challenge you solved.",
    "leadership_style": "How would you describe your leadership style?",
    "mentoring": "Have you mentored junior colleagues? How do you "
                 "approach it?",
    "agile": "What is your experience working in agile or scrum teams?",
    "code_quality": "How do you ensure the quality of your code?",
    "testing_approach": "How do you approach testing your work?",
    "estimation": "How do you estimate how long a piece of work will "
                  "take?",
    "ambiguity": "How do you handle ambiguity or unclear requirements?",
    "recent_learning": "What is something new you learned recently?",
    "proudest_moment": "What is the proudest moment of your career so far?",
    "why_industry": "Why do you want to work in this industry?",
    "handle_missed_deadline": "Tell me about a time you missed a deadline "
                              "and what you did about it.",
    "stakeholders": "How do you communicate technical topics to "
                    "non-technical stakeholders?",
}


def _store_dir() -> pathlib.Path:
    p = pathlib.Path("data") / "prepared"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _store_path(resume_id: str) -> pathlib.Path:
    safe = "".join(c for c in str(resume_id) if c.isalnum() or c in "-_")
    return _store_dir() / f"{safe}.json"


# In-memory cache: resume_id → (mtime, store_dict, matrix or None)
_CACHE: dict[str, tuple[float, dict, object]] = {}


def _norm_for_embed(text: str) -> str:
    """One normalization for BOTH store-time and ask-time embeddings —
    punctuation/case differences must never move the cosine."""
    out = []
    for ch in (text or "").lower():
        out.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(out).split())


def _cfg():
    from app.core.config_loader import cfg
    return cfg.live


def enabled() -> bool:
    return bool(getattr(_cfg(), "prepared_answers", True))


def threshold() -> float:
    return float(getattr(_cfg(), "prepared_match_threshold", 0.86))


# ---------------------------------------------------------------------------
# Generation (background, on resume upload)
# ---------------------------------------------------------------------------
async def prepare_for_resume(resume_id: str, profile: dict) -> int:
    """Generate + store prepared answers for `resume_id`. Returns how many
    were produced. Bounded concurrency; per-question fail-soft; embeddings
    stored alongside so matching needs one embed call at ask time."""
    if not enabled() or not resume_id or not isinstance(profile, dict) \
            or not profile:
        return 0
    try:
        from app.tools import persona_answer
    except Exception:  # noqa: BLE001
        return 0

    # FREE-TIER-SAFE pacing (2026-07-12: a resume upload rate-limited every
    # provider into hours-long cooldowns): strictly sequential, spaced calls,
    # and the whole batch aborts the moment the router reports exhaustion —
    # partial stores still serve; the rest generate on the next upload.
    sem = asyncio.Semaphore(1)
    pacing = float(getattr(_cfg(), "prepared_pacing_s", 2.0))
    exhausted = {"stop": False}
    answers: dict[str, dict] = {}
    count = int(getattr(_cfg(), "prepared_count", len(QUESTION_BANK)))
    items = list(QUESTION_BANK.items())[:max(1, count)]
    total = len(items)
    progressed = {"n": 0}

    def _report() -> None:
        # Surface generation as a live METRIC on the resume's progress entry
        # (post-finish safe): the upload modal / resume detail shows
        # "instant answers k/N" ticking without blocking anything.
        try:
            from app.documents import progress as _pg
            snap = _pg.get(str(resume_id)) or {}
            base = (snap.get("detail") or "").split(" · ")[0]
            detail = ((base + " · ") if base else "") + \
                f"instant answers {progressed['n']}/{total}"
            _pg.note_background(
                str(resume_id),
                counts={"instant_answers_done": progressed["n"],
                        "instant_answers_total": total},
                detail=detail)
        except Exception:  # noqa: BLE001
            pass

    async def _one(key: str, question: str) -> None:
        async with sem:
            if exhausted["stop"]:
                progressed["n"] += 1
                return
            try:
                text = await persona_answer.answer(
                    question=question, profile=profile,
                    qtype="behavioral", profile_q=True)
                if text and len(text.split()) >= 20:
                    answers[key] = {"question": question, "answer": text}
                if pacing > 0:
                    await asyncio.sleep(pacing)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "no llm route" in msg or "exhausted" in msg                         or "rate limit" in msg:
                    exhausted["stop"] = True
                    log.warning("prepared: providers exhausted — aborting "
                                "the batch (%d/%d done)", len(answers),
                                total)
                else:
                    log.info("prepared: '%s' skipped (%s)", key, exc)
            finally:
                progressed["n"] += 1
                _report()

    _report()
    await asyncio.gather(*(_one(k, q) for k, q in items))
    if not answers:
        return 0

    store: dict = {
        "version": 1,
        "resume_id": str(resume_id),
        "created_at": time.time(),
        "answers": answers,
    }
    # Question embeddings (best-effort): stored so ask-time matching is one
    # embed + a dot product. Absent embeddings → cue-based fallback matching.
    try:
        from app.rag import embedder as _emb
        if _emb.is_ready():
            keys = list(answers.keys())
            vecs = _emb.embed([_norm_for_embed(answers[k]["question"])
                               for k in keys])
            store["embedding_keys"] = keys
            store["embeddings"] = [list(map(float, v)) for v in vecs]
    except Exception:  # noqa: BLE001
        pass

    tmp = _store_path(resume_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(store), encoding="utf-8")
    os.replace(tmp, _store_path(resume_id))
    _CACHE.pop(str(resume_id), None)
    log.info("prepared: %d answers ready for resume %s",
             len(answers), resume_id)
    return len(answers)


def drop(resume_id: str) -> None:
    """Invalidate on resume delete/replace."""
    _CACHE.pop(str(resume_id), None)
    try:
        _store_path(resume_id).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def has_store(resume_id: str) -> bool:
    return _store_path(resume_id).is_file()


def _load(resume_id: str) -> tuple[dict, object] | None:
    """(store, question_matrix|None) with mtime-validated caching."""
    p = _store_path(resume_id)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    cached = _CACHE.get(str(resume_id))
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    try:
        store = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    mat = None
    try:
        if store.get("embeddings"):
            import numpy as np
            mat = np.asarray(store["embeddings"], dtype="float32")
    except Exception:  # noqa: BLE001
        mat = None
    _CACHE[str(resume_id)] = (mtime, store, mat)
    return store, mat


# ---------------------------------------------------------------------------
# Matching (ask time)
# ---------------------------------------------------------------------------
def match(resume_id: str, question: str) -> dict | None:
    """The prepared answer for `question`, or None. Embedding cosine against
    the stored bank questions (threshold-gated); falls back to normalized
    exact-question equality when the embedder is cold. Never raises."""
    try:
        if not enabled() or not resume_id:
            return None
        loaded = _load(str(resume_id))
        if loaded is None:
            return None
        store, mat = loaded
        answers: dict = store.get("answers") or {}
        if not answers:
            return None
        q = " ".join((question or "").lower().split())
        if len(q) < 8:
            return None

        # Exact/normalized hit is free.
        for key, entry in answers.items():
            if " ".join(entry["question"].lower().split()).rstrip("?.!") \
                    == q.rstrip("?.!"):
                return {"key": key, "score": 1.0, **entry}

        if mat is None:
            return None
        from app.rag import embedder as _emb
        if not _emb.is_ready():
            return None
        import numpy as np
        v = np.asarray(_emb.embed([_norm_for_embed(q)])[0], dtype="float32")
        sims = mat @ v
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score < threshold():
            return None
        key = (store.get("embedding_keys") or list(answers.keys()))[idx]
        entry = answers.get(key)
        if not entry:
            return None
        return {"key": key, "score": score, **entry}
    except Exception:  # noqa: BLE001 — fail-open to live generation
        return None


__all__ = ["QUESTION_BANK", "prepare_for_resume", "match", "drop",
           "has_store", "enabled"]
