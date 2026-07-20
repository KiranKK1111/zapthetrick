# Live audio on a remote backend — FE captures audio, streams to backend

**Status:** implemented; backend + FE-wiring verified in-repo, native Windows
loopback capture written-to-spec but **pending on-hardware verification**.

This document is the detailed reference for how the Live interview-assist
feature captures audio and gets it to the backend, covering **both** the
backend (FastAPI, on a RunPod pod) and the frontend (Flutter desktop/mobile).

---

## 1. Problem & motivation

Live interview assist needs to hear **two** distinct audio sources and treat
them differently:

| Source | What it is | How it's answered |
|---|---|---|
| **Interviewer** | the other party's voice, coming out of *your* speakers (Zoom/Teams/Meet) | **transcribed → classified → answered** |
| **Candidate** | your own microphone | **absorbed** (echo-skipped, never answered) — except in Solo mode |

On Windows the interviewer source is the **system loopback** (WASAPI "what you
hear"). When the backend ran **locally**, it captured that loopback itself with
`pyaudiowpatch`. Two things broke this once the backend moved to a **remote
RunPod pod**:

1. The pod has no access to your machine's audio devices — there is nothing on
   the pod to capture.
2. `pyaudiowpatch` is Windows-only; the pod runs Linux.

The microphone was never affected — the Flutter app already captured the mic
and streamed int16 PCM to the backend over the Live WebSocket, and that kept
working. **Only the interviewer (loopback) source needed a new home.**

**Solution (Part 2):** the Flutter desktop app captures **both** sources
locally and streams them over the one existing Live WebSocket, **role-tagged**,
so the backend routes each source correctly *without capturing anything itself*.

---

## 2. Architecture overview

```
 Windows desktop (Flutter)                          RunPod pod (FastAPI)
 ┌─────────────────────────────┐                    ┌───────────────────────────┐
 │ WASAPI loopback ─┐          │   binary WS frames │  /ws/live receive loop     │
 │ (interviewer)    ├─ prefix ─┼───────────────────▶│  fe_tagged? split role byte│
 │ mic (candidate) ─┘  0x00/01 │  [role][int16 PCM] │    0x00 → segmenter (answer)│
 │                             │  +{client_capture} │    0x01 → candidate_segmenter│
 │                             │                    │  → AudioStreamSegmenter     │
 │                             │◀───────────────────│  → VAD → STT → classify →   │
 │  token/transcript/meta/done │   JSON text frames │     answer (token stream)   │
 └─────────────────────────────┘                    └───────────────────────────┘
```

**Two capture topologies, chosen at runtime by backend location:**

- **Local backend** (`localhost`/`127.0.0.1`/`::1`): unchanged. The FE sends a
  `start_capture` control frame; the **server** captures loopback via
  `pyaudiowpatch`; the mic streams **untagged** PCM. (Proven path — preserved so
  a local dev/desktop setup keeps working exactly as before.)
- **Remote backend** (any other host, i.e. a pod): the FE captures **both**
  loopback + mic locally and streams them **role-tagged**; the server captures
  nothing.

The switch is a single getter, `_remoteBackend`, in the Live screen.

---

## 3. Wire protocol

All additions are **backward-compatible** — an untagged client (mobile mic-only,
or a local backend) behaves exactly as before.

### 3.1 WebSocket

- **URL:** `/ws/live?resume_id=&session_id=&mode=` (`mode` = `standard` | `solo`).
- **Scheme:** `http→ws`, `https→wss`; `localhost` is rewritten to `127.0.0.1`
  (uvicorn binds IPv4, and Windows resolves `localhost` to `::1` first).

### 3.2 Frames

| Direction | Kind | Payload |
|---|---|---|
| C→S | **binary (untagged)** | raw int16 LE PCM @16 kHz mono |
| C→S | **binary (tagged)** | `[1 role byte][int16 LE PCM @16 kHz mono]` |
| C→S | text control | `{"type": ...}` — see below |
| S→C | text | `ready`/`transcript`/`meta`/`token`/`done`/`skipped`/`capture`/`error`/`pong` (JSON) |

**Role byte (tagged frames only):**

| Byte | Role | Backend routing | Answered? |
|---|---|---|---|
| `0x00` | interviewer (loopback) | main `segmenter` | yes (Standard & Solo) |
| `0x01` | candidate (mic) | `candidate_segmenter` | no in Standard; yes in Solo |

### 3.3 New control message

```json
{"type": "client_capture", "tagged": true}
```

- Sent **once** after connect, before any tagged binary frame.
- Flips a per-connection flag so the receive loop interprets subsequent binary
  frames as role-tagged.
- `{"tagged": false}` (sent on stop) restores legacy interpretation.
- Server replies `{"type":"capture","state":"client"|"server"}`.

WebSocket ordering guarantees `client_capture` is processed before the first
tagged frame that follows it on the same socket.

---

## 4. Backend implementation (`zapthetrick_be`)

All in `app/api/routes_ws.py`. The STT/VAD/answer pipeline downstream is
**unchanged** — this only changes how bytes on the wire are split and which
segmenter they enter.

### 4.1 Two segmenters (already existed)

The `/ws/live` handler builds two `AudioStreamSegmenter`s:

- `segmenter` — the **main** one; its utterances are classified and answered.
- `candidate_segmenter` — absorbs the candidate's own voice (echo-skip); never
  produces an answer.

`solo_mode = (mode == "solo")` makes **every** source flow to the main
`segmenter` (so a tester is answered whether they speak into the mic or play a
recording through the speakers).

### 4.2 Frame split — `_split_fe_frame()`

A pure, unit-tested helper (so the off-by-one-prone wire format is testable
without a live socket):

```python
def _split_fe_frame(raw: bytes) -> tuple[int, bytes]:
    """(role, pcm_bytes). Empty frame → (0, b"") so it degrades to interviewer."""
    if not raw:
        return 0, b""
    return raw[0], raw[1:]
```

### 4.3 Receive loop — binary branch

```python
if msg.get("bytes") is not None:
    raw = msg["bytes"]
    _seg = None
    if capture_state.get("fe_tagged"):
        role, raw = _split_fe_frame(raw)
        # SOLO answers every source; STANDARD answers only the interviewer.
        _seg = segmenter if solo_mode else (
            candidate_segmenter if role == 1 else segmenter)
    chunk = _decode_pcm(raw)                      # int16 LE → float32 [-1,1]
    if chunk is not None:
        if _seg is None:                          # untagged legacy routing
            _seg = (segmenter if solo_mode else (
                    candidate_segmenter
                    if capture_state.get("task") is not None
                    else segmenter))
        await _seg.push(chunk)
```

Key point: **only the routing decision changed.** After `_seg.push(chunk)` the
audio flows through the identical `AudioStreamSegmenter → VAD endpoint → STT
(transcribe_with_confidence) → classify → answer` path that server-captured
audio uses. Odd-length PCM after the strip returns `None` from `_decode_pcm`
(never raises) so one malformed frame can't kill the socket.

### 4.4 Control handler — `client_capture`

```python
elif kind == "client_capture":
    capture_state["fe_tagged"] = bool(payload.get("tagged", True))
    await send({"type": "capture",
                "state": "client" if capture_state["fe_tagged"] else "server"})
```

`capture_state` is the same dict already threaded to the disconnect/cleanup
path, so the flag lives and dies with the connection.

### 4.5 PCM decode (unchanged)

```python
def _decode_pcm(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr if arr.size else None
```

The STT stack expects **1-D float32 mono @ 16 kHz** (`cfg.audio.sample_rate`).

### 4.6 Tests

`tests/test_live_transport.py`:

- `test_fe_frame_split_roles` — role byte split + `_decode_pcm` on the stripped
  body for both roles; empty-frame degradation.
- `test_fe_frame_odd_pcm_after_strip_is_rejected` — a malformed odd-length body
  returns `None` rather than raising.

Verified: `test_live_transport.py` + `test_live_dual_source.py` → **19 passed**.

---

## 5. Frontend implementation (`zapthetrick_fe`)

### 5.1 WS client — `lib/services/ws_client.dart`

```dart
static const int roleInterviewer = 0x00; // loopback → answered
static const int roleCandidate   = 0x01; // mic → absorbed

void sendPcmTagged(Uint8List pcm, {required int role}) {
  final framed = Uint8List(pcm.length + 1);
  framed[0] = role & 0xFF;
  framed.setRange(1, framed.length, pcm);
  _channel?.sink.add(framed);
}

void clientCapture({bool tagged = true}) =>
    sendText({'type': 'client_capture', 'tagged': tagged});
```

`sendPcm` (untagged) and `startCapture`/`stopCapture` (server-side) are kept for
the legacy paths.

### 5.2 Capture selection — `lib/screens/live_listen.dart`

```dart
bool get _captureBoth => !_mobile && !kIsWeb;   // desktop
bool get _remoteBackend {                        // backend not on this machine
  final host = Uri.tryParse(ApiService.defaultBaseUrl)?.host ?? '';
  return !(host == 'localhost' || host == '127.0.0.1' ||
           host == '::1' || host.isEmpty);
}
```

`_toggleListen()` (desktop branch):

```dart
if (_remoteBackend) {
  _ws?.clientCapture(tagged: true);
  await _startLoopback();          // interviewer → tagged 0x00; warns if unsupported
  await _startMic(tagged: true);   // candidate   → tagged 0x01
} else {
  _ws?.startCapture(source: 'system_loopback'); // server captures loopback
  await _startMic();                            // mic untagged
}
```

- **Mobile/web** stay mic-only and **untagged** (a browser can't grab loopback;
  on mobile the single mic *is* the interviewer, so untagged → main segmenter).

### 5.3 Mic capture (candidate)

`_startMic({bool tagged = false})` uses the `record` package
(`AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1`) — already the
exact wire format — and, when `tagged`, sends each chunk via
`sendPcmTagged(chunk, role: WsClient.roleCandidate)` instead of `sendPcm`.

### 5.4 Loopback capture (interviewer) — new module

Three files, selected by a conditional import so web/other OSes never pull
`dart:io`/`win32`:

| File | Role |
|---|---|
| `lib/live/loopback_capture.dart` | abstract `LoopbackCapture`, `LoopbackUnsupported`, `createLoopbackCapture()` (conditional import) |
| `lib/live/loopback_capture_stub.dart` | non-io platforms → `makeLoopbackCapture() => null` (mic-only fallback) |
| `lib/live/loopback_capture_io.dart` | dart:io: Windows → WASAPI FFI capturer; other desktop → null |

**Interface contract:** `start(onChunk)` delivers **int16 LE mono PCM @ 16 kHz**
— exactly what `sendPcmTagged` and the backend `_decode_pcm` expect. `stop()`
is idempotent and releases all native resources.

`_startLoopback()` in the Live screen:

```dart
final cap = createLoopbackCapture();
if (cap == null) { _emitWarning('System-audio capture isn’t available…'); return; }
await cap.start((pcm) => _ws?.sendPcmTagged(pcm, role: WsClient.roleInterviewer));
_loopback = cap;
```

On any failure it warns and the session continues **mic-only** (Solo mode still
lets you test with the mic). Teardown (`_teardownLoopback()`) is wired into
`_stopListening`, the `capture` error frame, and the socket-death path so the
native capture never stays hot.

### 5.5 WASAPI loopback internals — `loopback_capture_io.dart`

Pure Dart FFI over `win32` 5.15.0 COM bindings (no C++/CMake — `flutter run`
just works). Flow:

1. `CoInitializeEx(nullptr, COINIT_MULTITHREADED)` — MTA, correct for
   shared-mode capture from a polling loop.
2. `MMDeviceEnumerator.createInstance()` →
   `getDefaultAudioEndpoint(eRender, eConsole)` — the **render** endpoint;
   capturing *its* stream in loopback mode yields "what the speakers play".
3. `IMMDevice.activate(IID_IAudioClient, CLSCTX_ALL(=23))` → `IAudioClient`.
4. `getMixFormat()` → the only format shared-mode loopback accepts. Parse it:
   - `WAVE_FORMAT_IEEE_FLOAT` (0x0003) → float32; `WAVE_FORMAT_PCM` (0x0001) →
     int; `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) → read `SubFormat.Data1` (0x3=float).
   - Record `nChannels`, `nSamplesPerSec` (usually 48 000 or 44 100), bytes/sample.
5. `initialize(SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK=0x00020000, 200 ms buffer,
   0, mixFormat, null)`.
6. `getService(IID_IAudioCaptureClient)` → `IAudioCaptureClient`; `Start()`.
7. **Poll loop** (`Timer.periodic(8 ms)`): while `getNextPacketSize() > 0`:
   `getBuffer()` → if `AUDCLNT_BUFFERFLAGS_SILENT` emit zeros, else read
   `frames × channels` samples → **downmix to mono** (average channels, scale to
   [-1,1]) → **resample to 16 kHz** → int16 → `onChunk` → `releaseBuffer()`.

**Streaming resampler** (`_Resampler`): linear interpolation with fractional
phase + a small carry buffer preserved across callbacks, so chunk boundaries
don't click. Handles any input rate (48 k / 44.1 k → 16 k).

**Teardown:** cancel timer, `IAudioClient.Stop()`, `release()` each COM
interface (capture, client, device, enumerator), `free` the IID GUIDs,
`CoTaskMemFree` the mix format, `CoUninitialize`.

---

## 6. Behaviour matrix

| Platform | Backend | Interviewer captured by | Mic frames | Answered source(s) |
|---|---|---|---|---|
| Desktop | local | server (`pyaudiowpatch`) | untagged | interviewer (Std) / both (Solo) |
| Desktop | remote (pod) | **FE WASAPI loopback** | **tagged 0x01** | interviewer 0x00 (Std) / both (Solo) |
| Mobile/web | any | — (none) | untagged | the single mic |

---

## 7. Verification status

**Verified in-repo**
- Backend role-split wire format — unit tests, `pytest` green (19 passed).
- `routes_ws` imports cleanly; live-transport + dual-source suites green.
- FE compiles — `flutter analyze` on the 5 changed files: **no issues**.

**Needs a real Windows machine (cannot run native audio in this env)**
- `loopback_capture_io.dart`: WASAPI device activation, `GetMixFormat` parsing,
  the loopback poll loop, and the resampler. Written to the WASAPI shared-mode
  loopback spec against `win32` 5.15.0, but **not yet run on hardware**.
- End-to-end: interviewer (system audio) transcribed by the pod while the mic is
  absorbed in Standard; both answered in Solo.

### On-machine test steps
1. Set the pod URL on the startup screen; open **Live**; press **Listen** on
   Windows desktop.
2. Play audio through the speakers / take a call → transcribes as interviewer
   and gets answered.
3. Speak into the mic in **Standard** → absorbed; in **Solo** → answered.
4. If loopback can't open you get a clear warning and it continues mic-only.

---

## 8. Not yet implemented / follow-ups
- **Linux/macOS loopback** — `loopback_capture_io.dart` returns null off Windows
  (mic-only fallback). Linux would capture a PulseAudio/PipeWire `.monitor`
  source; planned follow-up.
- **Auto-reconnect + tagged mode** — after a socket drop the FE re-enters tagged
  mode on the next Listen press; a silent mid-session reconnect does not
  auto-restore tagged capture (acceptable for now).

## 9. File index

| Concern | File |
|---|---|
| Role split + routing + `client_capture` | `zapthetrick_be/app/api/routes_ws.py` |
| Backend tests | `zapthetrick_be/tests/test_live_transport.py` |
| Tagged send / control helper | `zapthetrick_fe/lib/services/ws_client.dart` |
| Capture selection + wiring | `zapthetrick_fe/lib/screens/live_listen.dart` |
| Loopback interface + factory | `zapthetrick_fe/lib/live/loopback_capture.dart` |
| Loopback stub (non-Windows) | `zapthetrick_fe/lib/live/loopback_capture_stub.dart` |
| Loopback WASAPI FFI (Windows) | `zapthetrick_fe/lib/live/loopback_capture_io.dart` |
