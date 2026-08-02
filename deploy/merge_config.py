"""Merge the shipped config defaults into the pod volume's config.yaml.

Why this exists
---------------
The entrypoint used to render `config.yaml` once, on a volume's first boot, and
skip it forever after. The failure that caused was silent and severe: every
config key added after a volume was created stayed invisible to that pod. A pod
could run `llm.routing.max_retries: 15` in the repo while actually using the
code default of 6 — and the whole `llm.local` block, which is what makes the
never-empty routing ladder reach the on-pod GPU model, was simply absent. The
model was installed, running, and unroutable.

The contract
------------
1. Start from the shipped defaults (`config.example.yaml`).
2. **Deep-merge** the volume's existing file on top. Leaf-by-leaf: a shallow
   update would let an OLD `llm:` block replace the whole new one and take every
   added key down with it, which is the exact bug being fixed.
3. Force **pod-shaped** settings afterwards — database, ports, sandbox backend.
   Those describe this container, not an operator preference, so a stale value
   on the volume must not survive.
4. Seed provider keys from env **only when the volume has none**, so a fresh
   volume can answer without the UI while a key set later in Settings is never
   clobbered.
5. Write atomically, keeping a backup.

This is a standalone module rather than a heredoc inside the entrypoint
specifically so it can be tested — the previous version could not be, and that
is a large part of why the breakage went unnoticed for so long.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` onto `base`, descending into nested dicts.

    `over` wins at the leaves. Descending is the whole point: a shallow update
    replaces an entire section, so an old `llm:` block from a volume would
    silently drop every key added to the defaults since.
    """
    out = dict(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_existing(path: str) -> tuple[dict, bool]:
    """Read the volume's config. Returns `(config, was_readable)`.

    A corrupt file must not brick the pod: it degrades to defaults, but the
    broken file is preserved so nothing the operator set is actually lost.
    """
    if not os.path.exists(path):
        return {}, True
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("top level is not a mapping")
        return data, True
    except Exception as exc:  # noqa: BLE001
        print(f"!! existing config unreadable ({exc}) — using defaults")
        try:
            shutil.copy2(path, f"{path}.corrupt.{int(time.time())}")
        except Exception:  # noqa: BLE001
            pass
        return {}, False


def apply_pod_settings(cfg: dict, *, pgpass: str, app_port: int) -> dict:
    """Infrastructure that describes THIS pod. Forced after the merge."""
    db = cfg.setdefault("database", {})
    db.setdefault("postgres", {}).update(
        host="127.0.0.1", port=5432, db="postgres",
        schema_name="zapthetrick", user="postgres", password=pgpass,
        enable_age=False)
    db.setdefault("cache", {})["url"] = "redis://127.0.0.1:6379"
    cfg.setdefault("vision", {})["mode"] = "local"
    # The pod has no docker daemon, so the sandbox runs against the toolchains
    # baked into the image.
    cfg.setdefault("sandbox", {}).update(backend="local", enabled=True)
    cfg.setdefault("server", {}).update(host="0.0.0.0", port=int(app_port))
    return cfg


def apply_env_keys(cfg: dict, env: dict) -> dict:
    """Seed provider keys from env ONLY when the volume has none.

    Env exists to make a fresh volume answerable without opening the UI. A key
    the operator later set in Settings must outrank it.
    """
    llm = cfg.setdefault("llm", {})
    for env_key, cfg_key in (("OPENROUTER_API_KEY", "openrouter_api_key"),
                             ("NVIDIA_API_KEY", "nvidia_api_key")):
        val = (env.get(env_key) or "").strip()
        if val and not str(llm.get(cfg_key) or "").strip():
            llm[cfg_key] = val
    return cfg


def apply_local_floor(cfg: dict, env: dict) -> dict:
    """Enable the on-pod local generation floor (§2.1 T4).

    Forced every boot, deliberately: an old volume that predates the
    `llm.local` block gets it now. Without this the ladder's T4 rung is inert
    and 'No LLM route available' remains reachable even with the model running.
    """
    if (env.get("LOCAL_LLM_ENABLED") or "") != "1":
        return cfg
    loc = cfg.setdefault("llm", {}).setdefault("local", {})
    loc.update(
        enabled=True,
        model_id=env.get("LOCAL_LLM_MODEL_ID") or "qwen3-4b-instruct",
        base_url="http://127.0.0.1:%s/v1" % (env.get("LOCAL_LLM_PORT") or "8081"),
    )
    small = (env.get("LOCAL_LLM_SMALL_MODEL_ID") or "").strip()
    if small:
        loc["small_model_id"] = small
    return cfg


def build(defaults: dict, existing: dict, env: dict) -> dict:
    """The whole merge, as a pure function — this is what the tests exercise."""
    cfg = deep_merge(defaults, existing)
    cfg = apply_pod_settings(
        cfg, pgpass=env.get("PGPASS", ""), app_port=env.get("APP_PORT", 8000))
    cfg = apply_env_keys(cfg, env)
    cfg = apply_local_floor(cfg, env)
    return cfg


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: merge_config.py <defaults.yaml> <target.yaml>")
        return 2
    src, dst = argv[1], argv[2]

    with open(src, "r", encoding="utf-8") as fh:
        defaults = yaml.safe_load(fh) or {}
    existing, readable = load_existing(dst)
    if readable and os.path.exists(dst):
        try:
            shutil.copy2(dst, dst + ".bak")
        except Exception:  # noqa: BLE001
            pass

    cfg = build(defaults, existing, dict(os.environ))

    # Temp file then atomic swap: a crash mid-write must never leave the pod
    # with half a config.
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    os.replace(tmp, dst)

    added = sorted(set(defaults) - set(existing))
    print(f"wrote {dst} ({len(cfg)} top-level sections"
          + (f"; new: {', '.join(added)}" if added else "") + ")")
    routing = (cfg.get("llm") or {}).get("routing") or {}
    local = (cfg.get("llm") or {}).get("local") or {}
    print("   llm.routing.max_retries =",
          routing.get("max_retries", "<absent>"))
    print("   llm.local.enabled       =", local.get("enabled", "<absent>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
