"""Pod config merge — the fix for the frozen-config failure.

The entrypoint used to render `config.yaml` once and skip it forever after. The
result was invisible: a pod ran `max_retries: 15` in the repo while actually
using the code default of 6, and the entire `llm.local` block was absent, so the
on-pod GPU model was installed, running, and **unroutable**. Nothing failed
loudly; the pod just answered worse than the source said it would.

That went unnoticed partly because the logic lived in a heredoc and could not be
tested. It is a module now, and these are the properties that matter:

* new default keys REACH an old volume (the whole point);
* operator settings SURVIVE (or the fix would be a data-loss bug);
* pod-shaped infrastructure is forced (a stale port bricks the pod);
* a corrupt file degrades instead of bricking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import merge_config as M  # noqa: E402

_ENV = {"PGPASS": "secret", "APP_PORT": 8000}


# ── deep_merge ──────────────────────────────────────────────────────────────

def test_deep_merge_descends_instead_of_replacing_sections():
    """The bug in one test.

    A shallow update lets an OLD `llm:` block replace the whole new one, taking
    every key added since down with it — which is exactly how `routing` and
    `local` went missing from live pods.
    """
    defaults = {"llm": {"routing": {"max_retries": 15}, "local": {"enabled": True},
                        "model": "new-default"}}
    existing = {"llm": {"model": "operator-pick"}}
    out = M.deep_merge(defaults, existing)
    assert out["llm"]["model"] == "operator-pick", "operator value must win"
    assert out["llm"]["routing"]["max_retries"] == 15, "new key must survive"
    assert out["llm"]["local"]["enabled"] is True


def test_deep_merge_leaves_win_at_every_depth():
    out = M.deep_merge(
        {"a": {"b": {"c": 1, "d": 2}}},
        {"a": {"b": {"c": 99}}},
    )
    assert out["a"]["b"] == {"c": 99, "d": 2}


def test_deep_merge_does_not_mutate_its_inputs():
    defaults = {"a": {"b": 1}}
    existing = {"a": {"c": 2}}
    M.deep_merge(defaults, existing)
    assert defaults == {"a": {"b": 1}} and existing == {"a": {"c": 2}}


def test_a_non_dict_override_replaces_wholesale():
    # An operator who set a list or scalar where defaults have a mapping meant
    # it; merging into it would produce something neither side asked for.
    out = M.deep_merge({"a": {"b": 1}}, {"a": "literal"})
    assert out["a"] == "literal"


# ── The realistic scenario ──────────────────────────────────────────────────

def test_an_old_volume_gains_routing_and_the_local_floor():
    """The actual reported failure: a pod whose config predates both blocks."""
    defaults = {
        "llm": {"routing": {"max_retries": 15, "enabled": True},
                "local": {"enabled": False, "model_id": "qwen3-4b-instruct"},
                "openrouter_api_key": ""},
        "voice": {"engine": "staged"},
    }
    old_volume = {"llm": {"openrouter_api_key": "sk-operator-key"}}

    cfg = M.build(defaults, old_volume, {
        **_ENV, "LOCAL_LLM_ENABLED": "1", "LOCAL_LLM_PORT": "8081"})

    # The keys that were missing are now present…
    assert cfg["llm"]["routing"]["max_retries"] == 15
    assert cfg["llm"]["local"]["enabled"] is True
    assert cfg["llm"]["local"]["base_url"] == "http://127.0.0.1:8081/v1"
    # …a whole new section arrived…
    assert cfg["voice"]["engine"] == "staged"
    # …and the operator's key was not touched.
    assert cfg["llm"]["openrouter_api_key"] == "sk-operator-key"


def test_operator_settings_outrank_defaults():
    cfg = M.build(
        {"llm": {"model": "shipped-default", "temperature": 0.3}},
        {"llm": {"model": "operator-chose-this"}},
        _ENV)
    assert cfg["llm"]["model"] == "operator-chose-this"
    assert cfg["llm"]["temperature"] == 0.3


# ── Pod-shaped settings are forced ──────────────────────────────────────────

def test_infrastructure_is_forced_over_a_stale_volume_value():
    """These describe the container, not a preference. A stale port or DB host
    surviving a merge would brick the pod."""
    cfg = M.build(
        {}, {"server": {"host": "127.0.0.1", "port": 1234},
             "database": {"postgres": {"host": "old-host", "port": 9999}}},
        {"PGPASS": "pw", "APP_PORT": 8080})
    assert cfg["server"] == {"host": "0.0.0.0", "port": 8080}
    assert cfg["database"]["postgres"]["host"] == "127.0.0.1"
    assert cfg["database"]["postgres"]["password"] == "pw"


def test_sandbox_backend_is_pinned_local_for_the_pod():
    """The pod has no docker daemon; a volume left on `docker` silently loses
    every sandbox verification."""
    cfg = M.build({}, {"sandbox": {"backend": "docker"}}, _ENV)
    assert cfg["sandbox"]["backend"] == "local"
    assert cfg["sandbox"]["enabled"] is True


# ── Env keys seed, never clobber ────────────────────────────────────────────

def test_env_seeds_a_key_when_the_volume_has_none():
    cfg = M.build({"llm": {"openrouter_api_key": ""}}, {},
                  {**_ENV, "OPENROUTER_API_KEY": "sk-from-env"})
    assert cfg["llm"]["openrouter_api_key"] == "sk-from-env"


def test_env_never_overwrites_a_key_set_in_settings():
    """Env exists to make a FRESH volume answerable. Letting it win would
    silently revert a key the operator changed in the UI."""
    cfg = M.build({"llm": {}}, {"llm": {"openrouter_api_key": "sk-from-ui"}},
                  {**_ENV, "OPENROUTER_API_KEY": "sk-from-env"})
    assert cfg["llm"]["openrouter_api_key"] == "sk-from-ui"


def test_a_blank_env_key_is_ignored():
    cfg = M.build({"llm": {"nvidia_api_key": "kept"}}, {},
                  {**_ENV, "NVIDIA_API_KEY": "   "})
    assert cfg["llm"]["nvidia_api_key"] == "kept"


# ── Local floor ─────────────────────────────────────────────────────────────

def test_the_local_floor_stays_off_when_not_requested():
    cfg = M.build({"llm": {"local": {"enabled": False}}}, {},
                  {**_ENV, "LOCAL_LLM_ENABLED": "0"})
    assert cfg["llm"]["local"]["enabled"] is False


def test_the_small_tier_is_registered_only_when_named():
    base = {**_ENV, "LOCAL_LLM_ENABLED": "1"}
    assert "small_model_id" not in M.build({}, {}, base)["llm"]["local"]
    with_small = M.build({}, {}, {**base, "LOCAL_LLM_SMALL_MODEL_ID": "qwen2.5-3b"})
    assert with_small["llm"]["local"]["small_model_id"] == "qwen2.5-3b"


# ── File handling ───────────────────────────────────────────────────────────

def test_a_corrupt_volume_config_degrades_to_defaults(tmp_path):
    """It must not brick the pod — and the broken file is preserved so nothing
    the operator set is actually lost."""
    bad = tmp_path / "config.yaml"
    bad.write_text("{{{ not: valid: yaml", encoding="utf-8")
    existing, readable = M.load_existing(str(bad))
    assert existing == {} and readable is False
    assert list(tmp_path.glob("config.yaml.corrupt.*")), "no copy of the bad file"


def test_a_config_that_is_not_a_mapping_is_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    existing, readable = M.load_existing(str(p))
    assert existing == {} and readable is False


def test_a_missing_file_is_a_clean_first_boot(tmp_path):
    existing, readable = M.load_existing(str(tmp_path / "nope.yaml"))
    assert existing == {} and readable is True


def test_end_to_end_writes_atomically_and_keeps_a_backup(tmp_path, monkeypatch):
    defaults = tmp_path / "config.example.yaml"
    defaults.write_text(yaml.safe_dump({
        "llm": {"routing": {"max_retries": 15}, "local": {"enabled": False}},
        "voice": {"engine": "staged"},
    }), encoding="utf-8")
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(
        {"llm": {"openrouter_api_key": "sk-keep-me"}}), encoding="utf-8")

    monkeypatch.setenv("PGPASS", "pw")
    monkeypatch.setenv("APP_PORT", "8000")
    monkeypatch.setenv("LOCAL_LLM_ENABLED", "1")
    assert M.main(["merge_config.py", str(defaults), str(target)]) == 0

    out = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert out["llm"]["routing"]["max_retries"] == 15
    assert out["llm"]["local"]["enabled"] is True
    assert out["llm"]["openrouter_api_key"] == "sk-keep-me"
    assert out["voice"]["engine"] == "staged"
    assert (tmp_path / "config.yaml.bak").exists(), "no recoverable backup"
    assert not (tmp_path / "config.yaml.tmp").exists(), "temp file left behind"


def test_merging_twice_is_idempotent(tmp_path, monkeypatch):
    """The entrypoint runs on EVERY boot now, so a restart must not drift."""
    defaults = tmp_path / "d.yaml"
    defaults.write_text(yaml.safe_dump(
        {"llm": {"routing": {"max_retries": 15}}}), encoding="utf-8")
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump({"llm": {"model": "pick"}}),
                      encoding="utf-8")
    monkeypatch.setenv("PGPASS", "pw")
    monkeypatch.setenv("APP_PORT", "8000")

    M.main(["m", str(defaults), str(target)])
    first = target.read_text(encoding="utf-8")
    M.main(["m", str(defaults), str(target)])
    assert target.read_text(encoding="utf-8") == first


# ── The real shipped defaults actually carry the fix ────────────────────────

def test_the_shipped_defaults_contain_the_keys_that_were_missing():
    """If config.example.yaml ever loses these, the merge faithfully propagates
    nothing and the original bug returns."""
    path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["llm"]["routing"]["max_retries"] >= 15
    assert "local" in cfg["llm"], "llm.local absent from the shipped defaults"
    assert "voice" in cfg


def test_the_entrypoint_actually_calls_the_merge():
    """A perfect merge script that nothing invokes fixes nothing."""
    sh = (Path(__file__).resolve().parents[1]
          / "deploy" / "runpod_entrypoint.sh").read_text(encoding="utf-8")
    assert "merge_config.py" in sh
    # …and the old skip-if-exists guard is genuinely gone.
    assert "preserving Settings/keys already configured" not in sh
