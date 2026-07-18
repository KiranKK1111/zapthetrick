"""On-demand AI runtime pack (lite-build support).

The LITE frozen backend ships WITHOUT torch / onnxruntime (they're 4.6 GB of
the 5.2 GB freeze — see Dockerfile-era measurements). Instead, this module
downloads the exact wheels once per machine into a persistent folder:

    %LOCALAPPDATA%\\ZapTheTrickAI\\runtime\\site-packages

and prepends it to ``sys.path`` at startup. The pack SURVIVES app updates and
reinstalls (it lives outside the install dir, like the HF model cache), so the
installer stays lightweight forever and torch is never re-downloaded.

Sources (official CDNs — nothing for us to host, resumable via HTTP Range):
  * torch          — download.pytorch.org (cu128 build, matching the freeze)
  * onnxruntime-gpu — the CUDA-12 wheel feed (matches torch cu12x; the default
                      PyPI 1.27 build links CUDA 13 with no Windows DLLs)
  * small pure deps — PyPI (sympy/networkx/… — torch's imports that the lite
                      freeze can't trace)

Progress is reported through ``app.models_warmup`` so the existing "Preparing
models" first-run screen shows a live "AI runtime (PyTorch + CUDA)" row with
download percentages — same UX as the model downloads.

Everything is fail-open: no network → the row shows the error, cloud chat
works regardless, and the next launch retries. Full builds (torch bundled)
short-circuit to a no-op.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
import threading
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"  # e.g. cp312

# Pinned to the versions the FULL freeze ships (see requirements/venv). If you
# bump torch in the venv, bump here too — the build script asserts they match.
TORCH_VERSION = "2.11.0+cu128"
ORT_VERSION = "1.27.0"

_TORCH_URL = (
    "https://download.pytorch.org/whl/cu128/"
    f"torch-{TORCH_VERSION.replace('+', '%2B')}-{_PY_TAG}-{_PY_TAG}-win_amd64.whl"
)
# PEP 503 simple index for the CUDA-12 onnxruntime-gpu build.
_ORT_INDEX = (
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/"
    "onnxruntime-cuda-12/pypi/simple/onnxruntime-gpu/"
)

# (pypi_name, version) — torch's / ort's import-time deps the lite freeze
# can't trace (they're only reachable through the excluded packages). Small
# (< 15 MB total); duplicates of anything already frozen are harmless (the
# pack dir simply shadows them with the same versions).
_PYPI_DEPS: list[tuple[str, str]] = [
    ("sympy", "1.14.0"),
    ("mpmath", "1.3.0"),
    ("networkx", "3.6.1"),
    ("jinja2", "3.1.6"),
    ("markupsafe", "3.0.3"),
    ("flatbuffers", "25.12.19"),
    ("filelock", "3.29.4"),
    ("fsspec", "2026.6.0"),
    ("typing_extensions", "4.15.0"),
]

_lock = threading.Lock()
_started = False


def pack_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") \
        or os.path.expanduser("~")
    return Path(base) / "ZapTheTrickAI" / "runtime"


def site_dir() -> Path:
    return pack_root() / "site-packages"


def activate() -> None:
    """Prepend the pack to sys.path (idempotent). Call BEFORE app imports."""
    d = str(site_dir())
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
        importlib.invalidate_caches()


def torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:  # noqa: BLE001 — a broken partial install reads as absent
        return False


def ort_present() -> bool:
    try:
        return importlib.util.find_spec("onnxruntime") is not None
    except Exception:  # noqa: BLE001
        return False


def needed() -> bool:
    """True when this process lacks the ML runtime (lite build, pack absent)."""
    activate()
    return not (torch_present() and ort_present())


# ── download machinery ────────────────────────────────────────────────────
def _fmt_mb(n: float) -> str:
    return f"{n / (1024 * 1024):.0f} MB" if n < 1024 ** 3 \
        else f"{n / (1024 ** 3):.2f} GB"


def _download(url: str, dest: Path, on_progress) -> None:
    """Stream `url` to `dest` with HTTP-Range resume."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 416:          # .part already complete
                tmp.rename(dest)
                return
            resp.raise_for_status()
            if resp.status_code != 206:
                start = 0                        # server ignored Range
            total = start + int(resp.headers.get("content-length", 0) or 0)
            mode = "ab" if (start and resp.status_code == 206) else "wb"
            got = start
            with open(tmp, mode) as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 512):
                    f.write(chunk)
                    got += len(chunk)
                    on_progress(got, total)
    tmp.rename(dest)


def _resolve_pypi_url(name: str, version: str) -> str | None:
    """Wheel URL from the PyPI JSON API — prefer any-py3, else cp-win_amd64."""
    import httpx

    r = httpx.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30.0)
    r.raise_for_status()
    files = r.json().get("urls", [])
    wheels = [f for f in files if f.get("filename", "").endswith(".whl")]
    for f in wheels:                             # pure-python first
        if "none-any" in f["filename"]:
            return f["url"]
    for f in wheels:                             # then our platform build
        fn = f["filename"]
        if _PY_TAG in fn and "win_amd64" in fn:
            return f["url"]
    return None


def _resolve_ort_url() -> str | None:
    """Parse the CUDA-12 simple index for our onnxruntime-gpu wheel."""
    import httpx

    r = httpx.get(_ORT_INDEX, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    pat = re.compile(r'href="([^"]+)"')
    want = f"onnxruntime_gpu-{ORT_VERSION}-{_PY_TAG}-{_PY_TAG}-win_amd64.whl"
    for href in pat.findall(r.text):
        if want in href:
            return href if href.startswith("http") else \
                _ORT_INDEX.rstrip("/") + "/" + href.lstrip("./")
    return None


def _install_wheel(whl: Path) -> None:
    """Unzip a wheel into the pack's site-packages (wheels are plain zips;
    console scripts aren't needed, so extraction is sufficient)."""
    site = site_dir()
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(site)


def _marker(name: str, version: str) -> Path:
    return pack_root() / f".done-{name}-{version.replace('+', '_')}"


def ensure(set_stage=None) -> bool:
    """Download + install everything missing. Blocking; returns success.

    `set_stage(detail: str)` receives human-readable progress lines (wired to
    the models_warmup row by the caller).
    """
    def _note(detail: str) -> None:
        if set_stage:
            try:
                set_stage(detail)
            except Exception:  # noqa: BLE001
                pass
        log.info("runtime-pack: %s", detail)

    activate()
    if torch_present() and ort_present():
        return True

    # (label, name, version, url_resolver)
    items: list[tuple[str, str, str, object]] = [
        *[(f"dependency {n}", n, v, lambda n=n, v=v: _resolve_pypi_url(n, v))
          for n, v in _PYPI_DEPS],
        ("PyTorch (CUDA)", "torch", TORCH_VERSION, lambda: _TORCH_URL),
        ("ONNX Runtime (GPU)", "onnxruntime-gpu", ORT_VERSION, _resolve_ort_url),
    ]
    tmp_dir = pack_root() / "tmp"
    ok = True
    for label, name, version, resolver in items:
        if _marker(name, version).exists():
            continue
        try:
            url = resolver()
            if not url:
                raise RuntimeError(f"no wheel URL found for {name}=={version}")
            whl = tmp_dir / url.split("/")[-1].split("?")[0]

            def _prog(got: int, total: int, label=label) -> None:
                if total:
                    _note(f"Downloading {label}… "
                          f"{_fmt_mb(got)} / {_fmt_mb(total)}")
                else:
                    _note(f"Downloading {label}… {_fmt_mb(got)}")

            _note(f"Downloading {label}…")
            _download(url, whl, _prog)
            _note(f"Installing {label}…")
            _install_wheel(whl)
            _marker(name, version).parent.mkdir(parents=True, exist_ok=True)
            _marker(name, version).touch()
            try:
                whl.unlink()
            except OSError:
                pass
        except Exception as exc:  # noqa: BLE001 — fail-open per item
            ok = False
            log.warning("runtime-pack: %s failed: %s", name, exc)
            _note(f"{label} failed: {str(exc)[:120]}")
    activate()
    return ok and torch_present() and ort_present()


def ensure_async_then(after) -> None:
    """Warm-up orchestrator: if the runtime pack is missing, download it first
    (reporting into the models_warmup checklist), THEN run `after()` (the
    model warm-ups, which need torch importable). No-op'ish when present."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

    from app import models_warmup as _mw

    def _run() -> None:
        if needed():
            _mw.register("runtime", "AI runtime (PyTorch + CUDA)", cached=False)
            _mw.set_stage("runtime", _mw.STAGE_LOADING, "Preparing download…")
            good = ensure(
                set_stage=lambda d: _mw.set_stage("runtime", _mw.STAGE_LOADING, d))
            _mw.set_stage(
                "runtime",
                _mw.STAGE_READY if good else _mw.STAGE_ERROR,
                "Ready" if good else
                "Download failed — voice features unavailable; retries next launch.",
            )
        try:
            after()
        except Exception:  # noqa: BLE001 — warm-up is best-effort
            log.exception("post-runtime warmup failed")

    threading.Thread(target=_run, name="runtime-pack", daemon=True).start()
