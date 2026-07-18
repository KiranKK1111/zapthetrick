"""Local-first HuggingFace loading helper.

`from_pretrained` on an ALREADY-DOWNLOADED model still fires an online HEAD
request to check for a newer revision. On a slow or blocked connection to
huggingface.co that request times out and retries 5x with backoff (~30s of
noise per model) before finally falling back to the cache — the "why is startup
slow / why all these ReadTimeout warnings" the user sees.

`load_local_first` tries the local cache FIRST (`local_files_only=True` — no
network at all), and only falls back to a normal networked load when the files
aren't cached yet (a genuine first download). So cached models load instantly
and silently, while downloading a newly-selected model still works.
"""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


def load_local_first(loader: Callable, repo: str, **kwargs):
    """Call `loader(repo, ...)` preferring the local cache; fall back to a
    networked load (first download) if the model isn't cached."""
    try:
        return loader(repo, local_files_only=True, **kwargs)
    except Exception as exc:  # noqa: BLE001 — not cached (or offline) → download
        log.info("hf: %s not in local cache (%s) — fetching", repo,
                 type(exc).__name__)
        return loader(repo, **kwargs)
