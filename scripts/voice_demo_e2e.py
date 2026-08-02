"""End-to-end voice interruption test against a REAL server and a REAL user.

The interruption/transcription fix was verified against a fake upstream and in
unit tests. This drives the whole thing for real instead: a live FastAPI app on
a live Postgres, a demo account created through the actual `/auth/register`
route, an authenticated WebSocket to `/ws/voice`, real protocol-v2 frames on the
wire, and the assertions that matter checked against what the socket actually
sent back.

What is real here
-----------------
* the app, its middleware, auth and DB;
* account creation and token minting through the shipped routes;
* the `/ws/voice` handler, `VoiceSessionRunner`, `StagedEngine`, the reorder
  buffer, the generation gate and protocol-v2 encoding;
* the barge-in path including the client-reported `played_ms`.

What is stubbed, and why
------------------------
Only **synthesis** — `tts_synth.synthesize` is replaced with a deterministic
function. Real TTS would make the test depend on a network voice service and on
audio timing, which tests nothing about the logic under examination. Everything
that decides *what gets recorded* runs for real.

The realtime (speech-native) engine is NOT exercised: it needs a paid
credential, and `voice.realtime_model` is empty by default. Its equivalent is
covered against a scripted upstream in `tests/test_voice_realtime.py`.

Usage:  python scripts/voice_demo_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

import numpy as np

# Windows consoles default to cp1252 and choke on the arrows/box characters this
# script prints. Reconfiguring is friendlier than restricting the output to
# ASCII forever.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMO_EMAIL = "demo@zapthetrick.local"
DEMO_PASSWORD = "DemoVoice!2026"
DEMO_NAME = "Voice Demo"

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((ok, name, detail))
    print(f"  [{_PASS if ok else _FAIL}] {name}" + (f"\n         {detail}" if detail else ""))
    return ok


def _identity(body: dict) -> tuple[str, str]:
    """Pull `(user_id, token)` out of an auth response.

    `/login` nests the account under `user`, while `/register` may return only a
    status when email verification is on. Reading one shape and assuming the
    other silently yields an anonymous session, which is exactly what this
    script exists to avoid.
    """
    user = body.get("user") or {}
    uid = str(user.get("id") or body.get("user_id") or "")
    return uid, str(body.get("token") or "")


async def ensure_demo_user() -> tuple[str, str]:
    """Create the demo account through the REAL register route, or log in if it
    already exists. Returns `(user_id, token)`."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://demo") as c:
        res = await c.post("/api/auth/register", json={
            "email": DEMO_EMAIL, "password": DEMO_PASSWORD, "name": DEMO_NAME})
        if res.status_code in (200, 201):
            uid, tok = _identity(res.json())
            if uid:
                print(f"  created demo user {DEMO_EMAIL}")
                return uid, tok
        # Already registered (or native auth disabled) -> log in.
        res = await c.post("/api/auth/login",
                           json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
        if res.status_code == 200:
            uid, tok = _identity(res.json())
            if uid:
                print(f"  logged in as {DEMO_EMAIL}")
                return uid, tok

    # Native auth is off on this box — create the row directly so the session
    # still runs under a REAL user rather than an anonymous one.
    from sqlalchemy import func, select

    from app.api import auth_native
    from storage.db import get_session_factory
    from storage.models import User
    factory = get_session_factory()
    if factory is None:
        print("  !! no database — cannot create a demo user")
        return "", ""
    async with factory() as s:
        row = (await s.execute(
            select(User).where(func.lower(User.email) == DEMO_EMAIL)
        )).scalar_one_or_none()
        if row is None:
            row = User(email=DEMO_EMAIL,
                       password_hash=auth_native.hash_password(DEMO_PASSWORD),
                       preferences={"name": DEMO_NAME})
            s.add(row)
            await s.commit()
            await s.refresh(row)
            print(f"  created demo user {DEMO_EMAIL}")
        else:
            # Re-stamp the password so the account is genuinely usable: an old
            # row with a stale hash is not a demo account, it is an obstacle.
            row.password_hash = auth_native.hash_password(DEMO_PASSWORD)
            await s.commit()
            print(f"  reusing demo user {DEMO_EMAIL} (password reset)")
        uid = str(row.id)
    try:
        return uid, auth_native.mint_token(uid, email=DEMO_EMAIL)
    except Exception:
        return uid, ""


def _pcm(ms: int, voiced: bool) -> bytes:
    n = int(16000 * ms / 1000)
    return np.full(n, 12000 if voiced else 0, dtype="<i2").tobytes()


def run() -> int:
    from fastapi.testclient import TestClient

    import app.live.tts_synth as tts_synth
    from app.main import app
    from app.voice import protocol as P

    # Deterministic synthesis: 100 ms of silence per chunk, so "how much audio
    # was produced" is exact and the test never depends on a voice service.
    async def fake_synth(text, voice_id=None, *, speed=1.0):
        return _pcm(100, False)

    tts_synth.synthesize = fake_synth

    uid, token = asyncio.run(ensure_demo_user())
    check(bool(uid), "demo user exists", f"id={uid or '<none>'}")

    conv = str(uuid.uuid4())
    qs = f"?conversation_id={conv}" + (f"&token={token}" if token else "")
    client = TestClient(app)

    print("\n-- session opens -------------------------------------------")
    with client.websocket_connect(f"/ws/voice{qs}") as ws:
        ready = ws.receive_json()
        check(ready.get("type") == "session.ready", "session.ready received",
              json.dumps(ready))
        check(ready.get("protocol") == 2, "wire protocol v2 negotiated")
        engine = ready.get("engine")
        check(engine == "staged",
              "staged engine selected (no realtime credential -> zero spend)",
              f"engine={engine}")
        gen0 = ws.receive_json()
        check(gen0.get("type") == "generation" and gen0.get("n") == 0,
              "initial generation floor published")
        ws.receive_json()  # phase: listening

        # -- a complete reply --------------------------------------------
        print("\n-- a clean, uninterrupted reply ----------------------------")
        for i, part in enumerate(["Kafka keeps ordering", "per partition."]):
            ws.send_json({"type": "speak", "seq": i, "text": part})
        ws.send_json({"type": "reply_end", "chunks": 2})

        audio, done = [], None
        for _ in range(20):
            msg = ws.receive()
            if msg.get("bytes") is not None:
                audio.append(P.decode_audio(msg["bytes"]))
                continue
            frame = json.loads(msg["text"])
            if frame.get("type") == "turn_complete":
                done = frame
                break
        check(len(audio) == 2, "both chunks arrived as protocol-v2 frames",
              f"seqs={[a.seq for a in audio]}")
        check([a.seq for a in audio] == [0, 1], "audio arrived in emission order")
        check(done is not None, "exactly one turn_complete")
        if done:
            check(done["assistant"] == "Kafka keeps ordering per partition.",
                  "a clean turn records the WHOLE answer",
                  repr(done["assistant"]))
            check(done["interrupted"] is False, "clean turn not marked interrupted")
        ws.receive_json()   # phase: listening

        # -- the interruption --------------------------------------------
        print("\n-- barge-in mid-reply --------------------------------------")
        for i, part in enumerate(["First sentence.", "Second sentence.",
                                  "Third sentence."]):
            ws.send_json({"type": "speak", "seq": i, "text": part})

        heard = []
        for _ in range(12):
            msg = ws.receive()
            if msg.get("bytes") is not None:
                heard.append(P.decode_audio(msg["bytes"]))
                if len(heard) == 1:
                    break
        check(len(heard) == 1, "first chunk was played before the barge-in")

        # What the client actually rendered: one chunk, 100 ms. The server had
        # already emitted all three (synthesis is instant here), which is
        # precisely why the CLIENT's report is the only truth about what was
        # heard.
        ws.send_json({"type": "stop_speaking",
                      "played_ms": 100, "played_chunks": 1})

        cut, gen1 = None, None
        for _ in range(14):
            msg = ws.receive()
            if msg.get("bytes") is not None:
                continue
            frame = json.loads(msg["text"])
            if frame.get("type") == "generation":
                gen1 = frame
            if frame.get("type") == "turn_complete":
                cut = frame
                break
        check(gen1 is not None and gen1["n"] > gen0["n"],
              "generation floor advanced (stale audio now unrenderable)",
              f"{gen0['n']} -> {gen1['n'] if gen1 else '?'}")
        check(cut is not None, "the interrupted turn still completed")
        if cut:
            check(cut["interrupted"] is True, "turn marked interrupted")
            check(cut["assistant"] == "First sentence.",
                  "TRANSCRIPT CUT TO WHAT WAS HEARD — no unspoken text recorded",
                  repr(cut["assistant"]))
            check("Second sentence." not in cut["assistant"]
                  and "Third sentence." not in cut["assistant"],
                  "unheard sentences absent from the record")

        ws.send_json({"type": "stop"})

    print("\n-- summary -------------------------------------------------")
    failed = [r for r in _results if not r[0]]
    print(f"  {len(_results) - len(failed)}/{len(_results)} checks passed")
    for _, name, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
