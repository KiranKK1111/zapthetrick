"""Shared pytest fixtures.

Semantic intent (bge-m3) is now the PRIMARY classifier and is ON in config, but
loading a ~2GB model on every `assess()`/`detect_intent_smart()` call would make
the unit suite slow and model-dependent. So we disable it for tests by default:
the deterministic regex FALLBACK is what the intent/clarify tests assert, and the
semantic path is validated separately (fake embedder in test_intent_semantic.py,
plus an opt-in real-model benchmark gated behind RUN_SEMANTIC_EVAL=1).
"""
from __future__ import annotations

import warnings

import pytest

# A SWIG-wrapped native dependency (praat-parselmouth, pulled in by the prosody
# analyzer) registers `swigvarlink` / `SwigPy*` types that lack `__module__` on
# Python 3.12, so CPython emits DeprecationWarnings — including at interpreter
# SHUTDOWN, which pytest.ini's session-scoped filter can't reach. Install a
# process-level, permanent filter at collection-import time so those never
# surface. They say nothing about our code or the tests.
for _swig in ("swigvarlink", "SwigPyPacked", "SwigPyObject"):
    warnings.filterwarnings(
        "ignore",
        message=rf"builtin type {_swig} has no __module__ attribute",
        category=DeprecationWarning,
    )


@pytest.fixture(autouse=True)
def _disable_semantic_intent(monkeypatch):
    try:
        from app.core import config_loader as C
        base = C.get_config()
        patched = base.model_copy(update={
            "semantic_intent": base.semantic_intent.model_copy(
                update={"enabled": False}),
            # config.yaml runs the sandbox on the DOCKER backend; unit tests
            # exercise the LOCAL executor + registry logic (no container in CI),
            # so force local here. The docker backend is integration-verified
            # against the real container (scripts/verify_sandbox_container.py).
            "sandbox": base.sandbox.model_copy(update={"backend": "local"}),
        })
        monkeypatch.setattr(C, "_config", patched)
    except Exception:  # noqa: BLE001 — never block collection on config issues
        pass
    # Process-global stores that tests mutate — clear them around every test so
    # one test's fake vectors / learned exemplars can't leak into another.
    _isolate_global_state()
    yield
    _isolate_global_state()


def _isolate_global_state() -> None:
    # `cfg` is a _CfgProxy that resolves sections via __getattr__ from the config
    # singleton. `monkeypatch.setattr(cfg, "<section>", fake)` leaves a SHADOWING
    # instance attribute after its revert (there was no real attr to restore to),
    # which then masks the real config in later tests. Drop any such shadows.
    try:
        from app.core.config_loader import cfg as _cfgproxy
        _cfgproxy.__dict__.clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.rag import embedder as _emb
        _emb._ONE_CACHE.clear()
        _emb._CACHE_ENABLED = False       # off in tests (isolation); on in prod
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.clarify import learned_exemplars as _le
        _le._reset_for_test()          # wipe _POS/_NEG/spaces, skip disk load
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.clarify import intent_semantic as _si
        _si.reset_cache()              # drop the exemplar matrix (rebuilt fresh)
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.understanding import understanding_pass as _up
        _up.reset_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.core import calibration as _cal
        _cal._reset_for_test()
    except Exception:  # noqa: BLE001
        pass
