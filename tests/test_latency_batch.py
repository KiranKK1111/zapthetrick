"""Latency batch (2026-07-11, items #1-#5) — exhaustive coverage:

  #1 STT model selection (endpoint, labels, key gating, runtime switch)
  #2 Pre-generated resume answers (bank, generation, matching, lifecycle)
  #3 Semantic answer cache (exact + embedding tier, scoping, eviction)
  #4 Perceived-speed defaults (speculation / drafting / answer cache ON)
  #5 Prompt trim (compact profile, single resume representation)

Deterministic bag-of-words embeddings stand in for bge-m3; the LLM is
stubbed. No model loads, no network, no DB.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient


def _fake_vec(text: str) -> list[float]:
    vec = [0.0] * 64
    for w in (text or "").lower().split():
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % 64] += 1.0
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _fake_embed(texts):
    return [_fake_vec(t) for t in texts]


def _warm_embedder(monkeypatch):
    import app.rag.embedder as emb
    monkeypatch.setattr(emb, "is_ready", lambda: True)
    monkeypatch.setattr(emb, "embed", _fake_embed)


def _cold_embedder(monkeypatch):
    import app.rag.embedder as emb
    monkeypatch.setattr(emb, "is_ready", lambda: False)
    monkeypatch.setattr(emb, "ensure_loading_in_background", lambda: None)


# ═══════════════════════════════════════════════════════════════════════
# #2 — Pre-generated resume answers
# ═══════════════════════════════════════════════════════════════════════
_PROFILE = {"name": "Kiran", "summary": "Senior backend engineer",
            "skills": ["python", "kafka"]}


@pytest.fixture()
def prepared_env(tmp_path, monkeypatch):
    """Isolated store dir + stubbed generator + warm fake embedder."""
    from app.core.config_loader import cfg
    from app.live import prepared
    monkeypatch.setattr(prepared, "_store_dir", lambda: tmp_path)
    monkeypatch.setattr(cfg.live, "prepared_pacing_s", 0.0, raising=False)
    prepared._CACHE.clear()

    async def fake_answer(*, question, profile, qtype="behavioral",
                          profile_q=False, **kw):
        return (f"Spoken answer to {question} grounded in "
                f"{profile.get('name')} profile with plenty of words to "
                "clear the twenty word minimum threshold easily and then "
                "some more.")

    import app.tools.persona_answer as pa
    monkeypatch.setattr(pa, "answer", fake_answer)
    _warm_embedder(monkeypatch)
    yield prepared
    prepared._CACHE.clear()


class TestQuestionBank:
    def test_bank_size_and_shape(self):
        from app.live.prepared import QUESTION_BANK
        assert len(QUESTION_BANK) >= 55
        assert len(set(QUESTION_BANK)) == len(QUESTION_BANK)
        for k, q in QUESTION_BANK.items():
            assert k and q and len(q) > 10, k

    def test_bank_covers_the_staples(self):
        from app.live.prepared import QUESTION_BANK
        blob = " ".join(QUESTION_BANK.values()).lower()
        for phrase in ("tell me about yourself", "strengths", "weaknesses",
                       "why should we hire you", "salary", "conflict",
                       "five years", "leaving"):
            assert phrase in blob, phrase


class TestPrepareForResume:
    def test_generates_and_stores(self, prepared_env):
        n = asyncio.run(prepared_env.prepare_for_resume("r1", _PROFILE))
        assert n >= 30
        assert prepared_env.has_store("r1")
        store = json.loads(
            prepared_env._store_path("r1").read_text(encoding="utf-8"))
        assert store["resume_id"] == "r1"
        assert len(store["answers"]) == n
        assert len(store["embeddings"]) == n
        assert len(store["embedding_keys"]) == n

    def test_disabled_flag_skips(self, prepared_env, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "prepared_answers", False,
                            raising=False)
        assert asyncio.run(
            prepared_env.prepare_for_resume("r2", _PROFILE)) == 0

    def test_empty_profile_skips(self, prepared_env):
        assert asyncio.run(prepared_env.prepare_for_resume("r3", {})) == 0
        assert asyncio.run(
            prepared_env.prepare_for_resume("r3", None)) == 0  # type: ignore

    def test_count_cap(self, prepared_env, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "prepared_count", 5, raising=False)
        n = asyncio.run(prepared_env.prepare_for_resume("r4", _PROFILE))
        assert n == 5

    def test_short_answers_dropped(self, prepared_env, monkeypatch):
        async def short_answer(**kw):
            return "Too short."
        import app.tools.persona_answer as pa
        monkeypatch.setattr(pa, "answer", short_answer)
        assert asyncio.run(
            prepared_env.prepare_for_resume("r5", _PROFILE)) == 0
        assert not prepared_env.has_store("r5")

    def test_one_failure_does_not_sink_the_rest(self, prepared_env,
                                                monkeypatch):
        calls = {"n": 0}

        async def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider hiccup")
            return ("A good long answer with more than twenty words in it "
                    "to satisfy the quality floor of the generator easily "
                    "and reliably every time.")
        import app.tools.persona_answer as pa
        monkeypatch.setattr(pa, "answer", flaky)
        n = asyncio.run(prepared_env.prepare_for_resume("r6", _PROFILE))
        assert n == len(prepared_env.QUESTION_BANK) - 1

    def test_exhaustion_aborts_batch(self, prepared_env, monkeypatch):
        # The 3rd call reports provider exhaustion → the batch STOPS
        # hammering; what was generated before still stores and serves.
        calls = {"n": 0}

        async def exhausted_after_two(**kw):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError(
                    "No LLM route available right now. All models exhausted.")
            return ("A perfectly fine long answer with easily more than "
                    "twenty words to clear the generator quality floor "
                    "without any trouble at all.")
        import app.tools.persona_answer as pa
        monkeypatch.setattr(pa, "answer", exhausted_after_two)
        n = asyncio.run(prepared_env.prepare_for_resume("rex", _PROFILE))
        assert n == 2                      # kept the two good ones
        assert calls["n"] == 3             # stopped after the failure
        assert prepared_env.has_store("rex")

    def test_cold_embedder_still_stores(self, prepared_env, monkeypatch):
        _cold_embedder(monkeypatch)
        n = asyncio.run(prepared_env.prepare_for_resume("r7", _PROFILE))
        assert n >= 30
        store = json.loads(
            prepared_env._store_path("r7").read_text(encoding="utf-8"))
        assert "embeddings" not in store


class TestPreparedMatch:
    def _ready(self, prepared_env, rid="m1"):
        asyncio.run(prepared_env.prepare_for_resume(rid, _PROFILE))
        return rid

    def test_exact_question_matches(self, prepared_env):
        rid = self._ready(prepared_env)
        hit = prepared_env.match(rid, "Tell me about yourself.")
        assert hit is not None
        assert hit["key"] == "about_yourself"
        assert hit["score"] == 1.0

    def test_normalized_exact(self, prepared_env):
        rid = self._ready(prepared_env)
        assert prepared_env.match(rid, "  tell me ABOUT yourself ") \
            is not None
        assert prepared_env.match(rid, "tell me about yourself?") is not None

    def test_semantic_paraphrase_matches(self, prepared_env):
        rid = self._ready(prepared_env)
        # Bag-of-words cosine: shares most words with the bank question.
        hit = prepared_env.match(rid, "about yourself tell me please")
        assert hit is not None and hit["key"] == "about_yourself"

    def test_unrelated_question_misses(self, prepared_env):
        rid = self._ready(prepared_env)
        assert prepared_env.match(
            rid, "explain kafka consumer group rebalancing") is None

    def test_threshold_respected(self, prepared_env, monkeypatch):
        rid = self._ready(prepared_env)
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "prepared_match_threshold", 1.01,
                            raising=False)
        assert prepared_env.match(
            rid, "about yourself tell me please") is None

    def test_short_question_never_matches(self, prepared_env):
        rid = self._ready(prepared_env)
        assert prepared_env.match(rid, "you?") is None

    def test_unknown_resume(self, prepared_env):
        assert prepared_env.match("nope", "Tell me about yourself.") is None

    def test_disabled_flag(self, prepared_env, monkeypatch):
        rid = self._ready(prepared_env)
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "prepared_answers", False,
                            raising=False)
        assert prepared_env.match(rid, "Tell me about yourself.") is None

    def test_cold_embedder_exact_still_works(self, prepared_env,
                                             monkeypatch):
        rid = self._ready(prepared_env)
        _cold_embedder(monkeypatch)
        assert prepared_env.match(rid, "Tell me about yourself.") is not None
        assert prepared_env.match(
            rid, "about yourself tell me please") is None   # semantic off

    def test_corrupted_store_fails_open(self, prepared_env):
        rid = self._ready(prepared_env)
        prepared_env._store_path(rid).write_text("{not json",
                                                 encoding="utf-8")
        prepared_env._CACHE.clear()
        assert prepared_env.match(rid, "Tell me about yourself.") is None

    def test_drop_removes_store(self, prepared_env):
        rid = self._ready(prepared_env)
        prepared_env.drop(rid)
        assert not prepared_env.has_store(rid)
        assert prepared_env.match(rid, "Tell me about yourself.") is None

    def test_store_rewrite_invalidates_cache(self, prepared_env):
        rid = self._ready(prepared_env)
        assert prepared_env.match(rid, "Tell me about yourself.") is not None
        import os
        import time
        p = prepared_env._store_path(rid)
        store = json.loads(p.read_text(encoding="utf-8"))
        store["answers"]["about_yourself"]["answer"] = "REWRITTEN ANSWER"
        p.write_text(json.dumps(store), encoding="utf-8")
        os.utime(p, (time.time() + 5, time.time() + 5))
        hit = prepared_env.match(rid, "Tell me about yourself.")
        assert hit is not None and hit["answer"] == "REWRITTEN ANSWER"

    def test_resume_id_path_sanitized(self, prepared_env):
        p = prepared_env._store_path("../../evil")
        assert ".." not in str(p.name)


# ═══════════════════════════════════════════════════════════════════════
# #3 — Semantic answer cache
# ═══════════════════════════════════════════════════════════════════════
class TestSemanticAnswerCache:
    def _cache(self):
        from app.perceived.cache import AnswerCache
        return AnswerCache()

    def test_exact_hit(self):
        c = self._cache()
        c.store("u1", "What is Kafka?", "Kafka is a log.")
        assert c.serve("u1", "What is Kafka?") == "Kafka is a log."

    def test_normalized_exact(self):
        c = self._cache()
        c.store("u1", "What is Kafka?", "Kafka is a log.")
        assert c.serve("u1", "  what   IS kafka? ") == "Kafka is a log."

    def test_semantic_hit_with_embed_fn(self):
        c = self._cache()
        c.store("u1", "how do i scale kafka consumers quickly",
                "Scale by partitions.",
                embedding=_fake_vec("how do i scale kafka consumers quickly"))
        got = c.serve("u1", "quickly scale kafka consumers how do i",
                      embed_fn=_fake_vec)
        assert got == "Scale by partitions."

    def test_semantic_below_threshold_misses(self):
        c = self._cache()
        c.store("u1", "how do i scale kafka consumers",
                "Scale by partitions.",
                embedding=_fake_vec("how do i scale kafka consumers"))
        assert c.serve("u1", "what is a binary tree",
                       embed_fn=_fake_vec) is None

    def test_scope_isolation(self):
        c = self._cache()
        c.store("u1", "what is kafka", "answer",
                embedding=_fake_vec("what is kafka"))
        assert c.serve("u2", "what is kafka", embed_fn=_fake_vec) is None

    def test_entries_without_embedding_skip_semantic(self):
        c = self._cache()
        c.store("u1", "what is kafka", "answer")     # no embedding stored
        assert c.serve("u1", "kafka what is it exactly",
                       embed_fn=_fake_vec) is None

    def test_validate_failure_invalidates(self):
        c = self._cache()
        c.store("u1", "what is kafka", "[LLM error: boom]")
        got = c.serve("u1", "what is kafka",
                      validate=lambda a: "[LLM error:" not in a)
        assert got is None
        # Entry was discarded — even a permissive validate finds nothing now.
        assert c.serve("u1", "what is kafka") is None

    def test_empty_or_low_quality_not_stored(self):
        c = self._cache()
        c.store("u1", "q", "")
        c.store("u1", "q2", "answer", quality_ok=False)
        assert c.serve("u1", "q") is None
        assert c.serve("u1", "q2") is None

    def test_embed_fn_error_fails_open(self):
        c = self._cache()
        c.store("u1", "what is kafka", "answer",
                embedding=_fake_vec("what is kafka"))

        def boom(_):
            raise RuntimeError("embedder died")
        assert c.serve("u1", "kafka question", embed_fn=boom) is None

    def test_lru_eviction(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.perceived, "predictive_cache_max_entries", 3,
                            raising=False)
        c = self._cache()
        for i in range(5):
            c.store("u1", f"question number {i}", f"answer {i}")
        assert c.serve("u1", "question number 0") is None    # evicted
        assert c.serve("u1", "question number 4") == "answer 4"


# ═══════════════════════════════════════════════════════════════════════
# #1 — STT model selection
# ═══════════════════════════════════════════════════════════════════════
class TestSttModelsEndpoint:
    """LOCAL-ONLY policy (2026-07-12): no cloud entries, no hint, one
    resident engine."""

    def _app(self) -> FastAPI:
        from app.api.routes_stt import router
        app = FastAPI()
        app.include_router(router)
        return app

    def test_only_local_models_listed(self):
        body = TestClient(self._app()).get("/api/stt/models").json()
        ids = [m["id"] for m in body["models"]]
        assert ids[:2] == ["parakeet", "qwen_asr"]
        whisper = [i for i in ids if i.startswith("faster_whisper::")]
        assert len(whisper) >= 5
        assert "faster_whisper::large-v3" in whisper
        assert all(m["kind"] == "local" for m in body["models"])
        assert "hint" not in body

    def test_labels_clear_and_distinct(self):
        body = TestClient(self._app()).get("/api/stt/models").json()
        labels = [m["label"] for m in body["models"]]
        assert len(set(labels)) == len(labels)
        blob = " ".join(labels)
        assert "Parakeet" in blob and "Qwen3-ASR" in blob \
            and "Hugging Face" in blob and "Whisper" in blob
        for m in body["models"]:
            assert m["detail"], m["id"]
            assert isinstance(m["downloaded"], bool)

    def test_no_cloud_ever(self):
        body = TestClient(self._app()).get("/api/stt/models").json()
        blob = " ".join(m["id"] + m["label"] for m in body["models"])
        for word in ("groq", "openai", "mistral", "cloud", "Groq",
                     "OpenAI", "Mistral"):
            assert word not in blob

    def test_active_simple_provider(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.stt, "provider", "qwen_asr", raising=False)
        body = TestClient(self._app()).get("/api/stt/models").json()
        assert body["active"] == "qwen_asr"
        active = [m for m in body["models"] if m["active"]]
        assert len(active) == 1 and active[0]["id"] == "qwen_asr"

    def test_active_composite_whisper_size(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.stt, "provider", "faster_whisper",
                            raising=False)
        monkeypatch.setattr(cfg.stt, "model", "small.en", raising=False)
        body = TestClient(self._app()).get("/api/stt/models").json()
        assert body["active"] == "faster_whisper::small.en"

    def test_downloaded_state_probed(self, monkeypatch):
        import app.api.routes_stt as rs
        monkeypatch.setattr(rs, "_downloaded",
                            lambda repo: repo is None or "parakeet" in
                            str(repo))
        body = TestClient(self._app()).get("/api/stt/models").json()
        by_id = {m["id"]: m for m in body["models"]}
        assert by_id["parakeet"]["downloaded"] is True
        assert by_id["qwen_asr"]["downloaded"] is False


class TestSttRuntimeSwitch:
    def test_provider_chain_rereads_config(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.stt import factory
        monkeypatch.setattr(cfg.stt, "provider", "parakeet", raising=False)
        monkeypatch.setattr(cfg.stt, "fallback_providers", [],
                            raising=False)
        assert factory._provider_chain() == ["parakeet"]
        monkeypatch.setattr(cfg.stt, "provider", "qwen_asr", raising=False)
        assert factory._provider_chain() == ["qwen_asr"]

    def test_no_fallback_when_cleared(self, monkeypatch):
        # The EXCLUSIVE-selection contract: an empty fallback list means the
        # chain is exactly the chosen engine — nothing else ever runs.
        from app.core.config_loader import cfg
        from app.stt import factory
        monkeypatch.setattr(cfg.stt, "provider", "faster_whisper",
                            raising=False)
        monkeypatch.setattr(cfg.stt, "fallback_providers", [],
                            raising=False)
        assert factory._provider_chain() == ["faster_whisper"]

    def test_no_cloud_dispatch_exists(self):
        from app.stt import factory
        assert not hasattr(factory, "_async_providers")
        import importlib.util
        assert importlib.util.find_spec("app.stt.cloud_stt") is None

    def test_unload_all_frees_engines(self, monkeypatch):
        from app.stt import factory, parakeet_stt, qwen_asr_stt, whisper_stt
        parakeet_stt._model_cache = object()          # pretend loaded
        cleared = {"qwen": 0, "whisper": 0}
        monkeypatch.setattr(qwen_asr_stt._model, "cache_clear",
                            lambda: cleared.__setitem__("qwen", 1))
        monkeypatch.setattr(whisper_stt._model, "cache_clear",
                            lambda: cleared.__setitem__("whisper", 1))
        factory.unload_all()
        assert parakeet_stt._model_cache is None
        assert cleared == {"qwen": 1, "whisper": 1}

    def test_settings_subscriber_unloads_and_warms(self, monkeypatch):
        from app.settings import subscribers
        from app.stt import factory
        called = {"unload": 0, "warm": 0}
        monkeypatch.setattr(factory, "unload_all",
                            lambda: called.__setitem__(
                                "unload", called["unload"] + 1))

        async def fake_warm():
            called["warm"] += 1
        monkeypatch.setattr(factory, "warm_active", fake_warm)

        async def go():
            await subscribers._on_stt("stt", {"provider": "qwen_asr"}, {})
            # let the fire-and-forget warm task run
            await asyncio.sleep(0)
        asyncio.run(go())
        assert called["unload"] == 1
        assert called["warm"] == 1

    def test_warm_active_fail_open(self, monkeypatch):
        from app.stt import factory

        async def boom(audio, prompt=None):
            raise RuntimeError("no model on this box")
        monkeypatch.setattr(factory, "transcribe_with_confidence", boom)
        asyncio.run(factory.warm_active())     # must not raise


class TestPerceivedDefaults:
    def test_defaults_on(self):
        from app.core.config_loader import Config
        p = Config().perceived
        assert p.speculation_enabled is True
        assert p.speculative_drafting is True
        assert p.answer_cache is True
        # Guards stay in place — and speculation is BUDGETED (2026-07-12:
        # unbounded racing exhausted free-tier keys).
        assert p.max_concurrent_drafts == 2
        assert p.cache_similarity_threshold == 0.95
        assert p.speculation_period_budget == 60

    def test_live_prepared_defaults(self):
        from app.core.config_loader import Config
        live = Config().live
        assert live.prepared_answers is True
        assert live.prepared_count == 64
        assert 0.5 < live.prepared_match_threshold < 1.0


# ═══════════════════════════════════════════════════════════════════════
# #5 — Prompt trim
# ═══════════════════════════════════════════════════════════════════════
class TestCompactProfile:
    def test_caps_and_drops(self):
        from app.persona.voice import _compact_profile
        prof = {
            "summary": "x" * 5000,
            "skills": [f"skill{i}" for i in range(60)],
            "projects": [{"name": "p", "desc": "y" * 900,
                          "_internal": "drop me"} for _ in range(20)],
            "raw_text": "the whole resume",
            "_analyzing": False,
            "name": "Kiran",
        }
        out = _compact_profile(prof)
        assert len(out["summary"]) == 1200
        assert len(out["skills"]) == 25
        assert len(out["projects"]) == 8
        assert len(out["projects"][0]["desc"]) == 300
        assert "_internal" not in out["projects"][0]
        assert "raw_text" not in out
        assert "_analyzing" not in out
        assert out["name"] == "Kiran"

    def test_fail_open_on_garbage(self):
        from app.persona.voice import _compact_profile
        assert _compact_profile("not a dict") == "not a dict"  # type: ignore

    def test_prompt_uses_compact(self):
        from app.persona.voice import build_interview_answer_prompt
        prof = {"summary": "s" * 5000, "raw_text": "SECRET_RAW"}
        p = build_interview_answer_prompt(prof, "technical_concept")
        assert "SECRET_RAW" not in p
        assert "s" * 1201 not in p


class TestSingleRepresentation:
    def test_profile_question_gets_profile_no_context(self):
        from app.tools.persona_answer import _build_messages
        msgs = _build_messages(
            question="Tell me about yourself",
            profile=_PROFILE, context=None, prior_qa=None,
            qtype="behavioral", profile_q=True)
        sys = msgs[0]["content"]
        user = msgs[1]["content"]
        assert "Kiran" in sys                      # profile embedded once
        assert "ADDITIONAL CONTEXT" not in user    # no second copy

    def test_non_profile_with_empty_profile_has_no_block(self):
        from app.tools.persona_answer import _build_messages
        msgs = _build_messages(
            question="What is Kafka?", profile={}, context=None,
            prior_qa=None, qtype="technical_concept", profile_q=False)
        assert "CANDIDATE PROFILE" not in msgs[0]["content"]

    def test_answer_accepts_profile_q(self):
        import inspect

        from app.tools import persona_answer
        assert "profile_q" in inspect.signature(
            persona_answer.answer).parameters
