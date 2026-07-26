"""Provider-error classification for gone / end-of-life models.

Regression guard for the NVIDIA NIM 410 "the model '…' has reached its end of
life … and is no longer available" error surfacing to the user instead of
failing over. A 410 (and EOL/decommissioned message markers) must be BOTH
retryable (fall over to the next free model) AND permanent_dead (pruned from the
catalog so it never wastes another routing attempt) — while transient 404s and
still-usable "deprecated" models stay alive.
"""
import pytest

from app.llm.providers.base import classify_error

_NIM_410 = (
    "NVIDIA NIM API error 410: {\"type\":\"about:blank\",\"title\":\"Gone\","
    "\"status\":410,\"detail\":\"The model 'qwen/qwen3.5-122b-a10b' has reached "
    "its end of life on 2026-07-20T00:00:00Z and is no longer available.\"}"
)


@pytest.mark.parametrize("status,msg,retryable,dead", [
    (410, _NIM_410, True, True),                         # the reported error
    (410, "gateway gone", True, True),                   # bare 410 Gone
    (400, "xyz is not a valid model ID", True, True),    # invalid id
    (404, "No endpoints found for abc", True, True),     # OpenRouter dead id
    (404, "page not found", True, False),                # transient CDN 404
    (200, "note: model is deprecated but still usable", False, False),  # keep it
    (429, "too many requests", True, False),             # rate limit → retry
    (None, "connection reset", True, False),             # transport → retry
    (503, "service unavailable", True, False),
])
def test_classify(status, msg, retryable, dead):
    assert classify_error(status, msg) == (retryable, dead)
