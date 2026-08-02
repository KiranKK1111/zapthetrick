"""Live-module isolation contract (design §L7).

These are structural checks, not behavioural ones. They exist because "we were
careful" is not a guarantee: the whole point of moving voice out of the Live
file is that a future edit should be *incapable* of regressing Live, and only a
test can hold that line once the author has moved on.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

_BE = pathlib.Path(__file__).resolve().parents[1]
_LIVE = _BE / "app" / "live"
_ROUTES_WS = _BE / "app" / "api" / "routes_ws.py"


def _py_files(root: pathlib.Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


# ── L1: physical separation ──────────────────────────────────────────────────

def test_voice_endpoint_is_not_in_the_live_file():
    """`/ws/voice` must not be declared in the file that serves `/ws/live`."""
    src = _ROUTES_WS.read_text(encoding="utf-8")
    assert '@router.websocket("/ws/voice")' not in src
    assert '@router.websocket("/ws/live")' in src


def test_voice_endpoint_lives_in_its_own_module():
    from app.api import routes_voice_ws
    assert hasattr(routes_voice_ws, "voice_ws")
    src = pathlib.Path(inspect.getfile(routes_voice_ws)).read_text("utf-8")
    assert '@router.websocket("/ws/voice")' in src
    assert '/ws/live' not in src.replace("`/ws/live`", "")  # only in prose


def test_both_endpoints_are_registered():
    """The relocation must not have dropped a route."""
    from app.main import app
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/ws/voice" in paths
    assert "/ws/live" in paths


# ── L3: import direction is one-way ──────────────────────────────────────────

def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every module name this file actually imports.

    Parsed with AST rather than grepped, so a docstring that *mentions*
    `app.voice` (s2s.py's deprecation notice does, deliberately) is not
    mistaken for a dependency. Only real import statements count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_live_package_never_imports_voice():
    """`app/voice/` may import `app/live/`. The reverse is forbidden — that is
    what keeps voice changes structurally unable to reach Live."""
    offenders = []
    for path in _py_files(_LIVE):
        for mod in _imported_modules(path):
            if mod == "app.voice" or mod.startswith("app.voice."):
                offenders.append(f"{path.name}: imports {mod}")
    assert not offenders, "app/live/ must not import app.voice:\n" + \
        "\n".join(offenders)


def test_live_endpoint_file_imports_only_the_turn_gate_alias():
    """`routes_ws.py` keeps one deprecated alias that delegates into
    `app.voice.turn_gate`. That single reference is permitted (it is inside a
    function body, so no Live code path triggers it); anything else is not."""
    voice_imports = {m for m in _imported_modules(_ROUTES_WS)
                     if m == "app.voice" or m.startswith("app.voice.")}
    assert voice_imports <= {"app.voice.turn_gate"}, \
        f"unexpected app.voice imports in the Live file: {voice_imports}"


# ── L2: shared modules keep their signatures ────────────────────────────────

@pytest.mark.parametrize("module,func,params", [
    ("app.live.barge_in", "classify_utterance", ["text"]),
    ("app.live.hypothesis", "completeness", ["text"]),
    ("app.live.tts_synth", "synthesize", ["text"]),
])
def test_shared_module_signatures_are_stable(module, func, params):
    """Voice consumes these read-only. They may GAIN optional parameters; the
    existing leading ones must not move or be renamed (rule L2)."""
    mod = __import__(module, fromlist=[func])
    sig = inspect.signature(getattr(mod, func))
    names = list(sig.parameters)
    assert names[:len(params)] == params


def test_audio_segmenter_accepts_the_kwargs_voice_passes():
    """The staged engine wires the SHARED segmenter with these callbacks. If a
    Live-side refactor renames one, voice must fail here rather than at runtime
    on a user's microphone."""
    from app.audio.stream import AudioStreamSegmenter
    names = set(inspect.signature(AudioStreamSegmenter).parameters)
    for kw in ("on_utterance", "on_partial", "prompt_provider",
               "on_stt_status", "on_speech_start"):
        assert kw in names, f"segmenter lost the {kw} kwarg"


# ── L5: config namespacing ──────────────────────────────────────────────────

def test_voice_code_reads_no_live_config():
    """All new keys live under `voice.*`. Voice must never read `cfg.live`."""
    offenders = []
    for path in _py_files(_BE / "app" / "voice"):
        for line in path.read_text("utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            if "cfg.live" in s:
                offenders.append(f"{path.name}: {s}")
    assert not offenders, "voice must not read live config:\n" + \
        "\n".join(offenders)


# ── Requirement 11.1: no new boot-time model load ───────────────────────────

def test_importing_voice_loads_no_model():
    """Importing the package must not pull torch/onnx or open a socket. The
    realtime engine is network-only and the staged engine reuses the already
    resident STT instance.

    Run in a SUBPROCESS. An earlier version deleted `app.voice.*` from
    `sys.modules` in-process to force a fresh import — which silently corrupted
    module identity for every later test: `app.voice.realtime` was re-imported
    as a new object while other test modules still held the old one, so
    monkeypatching `app.voice.tools.dispatch` stopped reaching the code under
    test. A fresh interpreter is both safe and a stronger assertion.
    """
    import subprocess
    import sys
    code = (
        "import sys\n"
        "import app.voice.engine, app.voice.policy, app.voice.protocol,"
        " app.voice.turn_gate\n"
        "heavy = {'torch', 'onnxruntime', 'transformers',"
        " 'sentence_transformers', 'kokoro'}\n"
        "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy)\n"
        "print(','.join(found))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(_BE),
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = [m for m in out.stdout.strip().split(",") if m]
    assert not loaded, f"voice import pulled heavy deps: {loaded}"
