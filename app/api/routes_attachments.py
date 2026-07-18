"""Chat with attachments — multipart upload + SSE stream.

`POST /api/chat/upload-stream` accepts a message plus files. Documents (pdf,
docx, xlsx, json, md, txt, csv) are parsed to text and inlined into the model
context when they fit, or retrieved (top-k chunks) when oversized. Images
(png/jpeg/…) are sent straight to a vision model. Everything routes through the
`auto` LLM client: image-bearing turns go to a vision-capable model and large
inlined docs to a model whose context window fits, both with rank-based
fallback (see app/llm/engine.py + router.py).

Modeled on /api/chat/stream so persistence + SSE events match exactly.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm_client import LLMError, llm
from app.database import Conversation, Message, get_session
from app.documents.parser import (
    MAX_UPLOAD_BYTES,
    PasswordRequired,
    UnsupportedDocument,
    extract_document_text,
    is_image,
)
from app.pipeline import plan

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# Vision→task pipeline (Stage 1). A vision model READS the screenshot/image and
# returns a thorough, task-focused TEXT description; the actual task is then run
# by a SECOND model routed by difficulty (so a heavy task can use a strong,
# possibly non-vision model). This separates perception from execution — the
# vision model perceives, the task model reasons.
_VISION_SYSTEM = (
    "You are a precise vision analyst. Examine the attached image(s) and produce "
    "a thorough, structured, factual description focused on the user's request. "
    "Extract ALL relevant detail: visible text VERBATIM, UI layout and "
    "components, any code, table/data values, charts/diagrams, colours and "
    "styling, structure, and visible errors. Do NOT attempt the task yourself — "
    "only describe what is in the image(s) precisely enough that another model "
    "can act on it without seeing the image. Be complete but concise.\n\n"
    "IF the image shows a CODING PROBLEM or a code editor/IDE (e.g. LeetCode, "
    "an online judge, VS Code), you MUST report, explicitly and first:\n"
    "- PROGRAMMING LANGUAGE: the language selected in the editor — read the "
    "language dropdown/selector (often at the top of the code panel), the file "
    "extension, or infer it from the visible code's syntax. State it by name "
    "(e.g. 'Editor language: Dart'). Do NOT assume Python.\n"
    "- The full problem statement and every example/constraint, VERBATIM.\n"
    "- The starter code / function signature, transcribed VERBATIM (so the "
    "solution can match the exact class/method/signature shown)."
)


_VISION_REFUSAL_RE = re.compile(
    r"\b(can(?:no|')?t|cannot|unable to|not able to|don'?t have (?:the )?"
    r"(?:ability|access))\b[^.\n]{0,45}\b(view|see|access|read|open|process|"
    r"display|analyz|interpret|render)\w*\b[^.\n]{0,30}\b(image|picture|photo|"
    r"screenshot|attachment|file|visual|it directly)\b",
    re.IGNORECASE,
)


async def _vision_extract(images: list[str], task: str) -> tuple[str, str, str]:
    """Stage 1: a vision model describes the image(s) so a text model can act.

    Returns `(analysis, ocr_text, vlm_text)`: `analysis` is the combined text for
    the SOLVER; `ocr_text` is the PURE local-OCR read; `vlm_text` is the PURE
    vision-model description. The two are returned SEPARATELY so language
    detection can run on the uncontaminated OCR text (a small VLM hallucinates a
    solution in the wrong language, and detecting on the merged string let that
    win over the real code).

    LOCAL-FIRST (VisionAnalysis.md): when `cfg.vision.enabled`, a LOCAL vision
    model (Qwen2.5-VL / MiniCPM-V, selectable like STT) parses the image to
    structured text — no API/provider vision model is used. Cached per image.
    Only if the local layer is disabled do we fall back to the legacy remote
    vision path below.

    Retries across DIFFERENT vision models if one refuses ("I can't view the
    image") or returns nothing/garbage — some provider endpoints silently drop
    the image. Returns "" if none worked (caller then attaches the image to the
    answer model directly, single-stage)."""
    # --- Local Vision Intelligence Layer (universal, offline; NO API vision) ---
    try:
        from app.core.config_loader import cfg
        _local_enabled = bool(getattr(cfg.vision, "enabled", True))
    except Exception:  # noqa: BLE001
        _local_enabled = True
    if _local_enabled:
        # The local model is a task-agnostic PARSER; steer it lightly with the
        # user's request so it foregrounds the relevant detail. For a CODE
        # screenshot use the STRICT read-only prompt (name the selected language
        # + transcribe verbatim; never solve) and do NOT append the solve
        # directive — that would coax the model into writing a solution, which is
        # exactly what makes it emit the wrong language.
        try:
            from app.codeintel.code_language import looks_like_coding_problem
            _is_code_img = looks_like_coding_problem(task or "")
        except Exception:  # noqa: BLE001
            _is_code_img = False
        _code_prompt = str(getattr(cfg.vision, "code_prompt", "") or "")
        _prompt = (_code_prompt if (_is_code_img and _code_prompt)
                   else cfg.vision.prompt)
        if not _is_code_img and (task or "").strip():
            _prompt = f"{_prompt}\n\nUser's request (for focus): {task.strip()}"
        local = ""
        ocr_text = ""
        try:
            from app.vision import factory as _vf
            from app.vision import ocr as _ocr
            # VLM (structure/understanding) + OCR (exact text) run CONCURRENTLY
            # — total latency is the slower of the two, not the sum. OCR reads
            # the dense code + the selected-language chip that a small VLM skips.
            _vlm_res, _ocr_res = await asyncio.gather(
                _vf.describe_images(images, _prompt),
                _ocr.ocr_images(images),
                return_exceptions=True,
            )
            local = (_vlm_res if isinstance(_vlm_res, str) else "").strip()
            ocr_text = (_ocr_res if isinstance(_ocr_res, str) else "").strip()
        except Exception as exc:  # noqa: BLE001 — never break the turn over vision
            log.info("vision-extract: local vision layer error (%s)", exc)
        _vlm_ok = bool(local) and len(local) > 30 and not _VISION_REFUSAL_RE.search(local)
        _ocr_block = ("[Exact text read from the image (OCR — authoritative, "
                      "trust this over the description below)]:\n" + ocr_text[:6000]
                      if ocr_text else "")
        if _vlm_ok and _ocr_block:
            # OCR FIRST + marked authoritative: a small VLM can paraphrase or
            # hallucinate dense content, so the solver must defer to the exact
            # OCR text; the VLM adds visual/layout context only.
            log.info("vision-extract OK (OCR %d + VLM %d chars)",
                     len(ocr_text), len(local))
            return (f"{_ocr_block}\n\n[Rough visual description (may paraphrase "
                    f"or err — defer to the exact text above)]:\n{local}",
                    ocr_text, local)
        if ocr_text:
            log.info("vision-extract OK via OCR only (%d chars)", len(ocr_text))
            return (_ocr_block, ocr_text, local)
        if _vlm_ok:
            log.info("vision-extract OK via LOCAL VLM (%d chars)", len(local))
            return (local, "", local)
        # Local IS the policy — never fall back to a remote/API vision model.
        log.info("vision-extract: local model unavailable/empty — NOT using a "
                 "remote vision model (local-only image reading)")
        return ("", ocr_text, local)

    # Local layer disabled → legacy remote vision path (below).
    sys_user = {
        "role": "user",
        "content": (
            f"The user's request about these image(s): "
            f"{(task or '').strip() or 'Describe them in detail.'}\n\n"
            "Describe the image(s) now."
        ),
        "images": images,
    }
    msgs = [{"role": "system", "content": _VISION_SYSTEM}, sys_user]
    avoid = None
    for attempt in range(3):
        try:
            # Bound each vision call: a stalled provider must not hang the whole
            # turn (the keepalive loop would otherwise keep the socket open
            # forever with no answer — the "no response" the user saw). On
            # timeout we fall through to attaching the image to the answer model.
            text, model_db_id = await asyncio.wait_for(
                llm.complete_routed(
                    msgs, None,
                    {"difficulty": "standard", "avoid_model_db_id": avoid},
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            log.info("vision-extract attempt %d timed out (45s)", attempt + 1)
            return ("", "", "")
        except Exception as exc:  # noqa: BLE001
            log.info("vision-extract attempt %d failed: %s", attempt + 1, exc)
            return ("", "", "")
        text = (text or "").strip()
        if text and len(text) > 30 and not _VISION_REFUSAL_RE.search(text):
            log.info("vision-extract OK (model_db_id=%s, %d chars)",
                     model_db_id, len(text))
            return (text, "", text)
        log.info("vision-extract: model_db_id=%s refused/empty — retrying "
                 "another vision model", model_db_id)
        if model_db_id is None:
            break
        avoid = model_db_id
    return ("", "", "")


async def _keepalive_until(task):
    """Yield SSE `: keepalive` comments every 8s while `task` runs, so a long
    await (vision extract / OCR / sandbox verify) never leaves the socket idle.
    The client's SSE parser has a ~30s inter-event watchdog that closes the
    stream on silence — without these, a slow vision read finishes on the
    backend but the app has already timed out and shows an EMPTY bubble. Ends
    when the task completes; the caller then reads task.result()."""
    while True:
        done, _pending = await asyncio.wait({task}, timeout=8.0)
        if done:
            return
        yield ": keepalive\n\n"


async def _vision_probe_language(images: list[str]) -> str:
    """Read the language selected in a coding screenshot's editor.

    OCR-FIRST: a small local VLM HALLUCINATES a Python/JS solution when it
    transcribes a coding screenshot, so its language guess is unreliable (the
    same Erlang problem came back as JS one run, Python the next). RapidOCR does
    exact character recognition — it reads the real "Erlang"/"Java" chip off the
    editor header and never invents a language. We return the OCR text (the
    caller runs `detect_language` on it — the "<Lang> Auto" chip + the code
    stub's own syntax resolve it). Falls back to a focused VLM prompt only when
    OCR is unavailable/empty. Local-only, never raises."""
    try:
        from app.vision import ocr as _ocr
        if _ocr.is_available():
            txt = await _ocr.ocr_images(images)
            if txt and txt.strip():
                return txt
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.vision, "enabled", True)):
            return ""
        from app.vision import factory as _vf
        probe = await _vf.describe_images(
            images,
            "This is a screenshot of a coding site (LeetCode/HackerRank) or an "
            "IDE. Look at the code editor panel and its header. What programming "
            "language is selected? Reply with ONLY the language name INCLUDING "
            "any version (e.g. Java, Python3, Python, C++, C#). If a code snippet "
            "is visible, also quote its first line verbatim.")
        return probe or ""
    except Exception:  # noqa: BLE001
        return ""


# Visually-exact intents: the user wants the model to reproduce the image
# precisely (clone/replicate a UI, pixel-perfect, match the design). For these
# we KEEP the image on the task model (a text description would lose fidelity)
# rather than handing off to a text-only model.
import re as _re_attach  # noqa: E402

_VISUAL_EXACT_RE = _re_attach.compile(
    r"\b(pixel[- ]?perfect|clone|replicate|recreate|reproduce|exact(?:ly)?|"
    r"match (?:the|this)|identical|same (?:design|layout|ui)|"
    r"convert (?:this|the) (?:design|screenshot|image|ui|mockup))\b",
    _re_attach.IGNORECASE,
)

# A zip/archive/download request packages an existing deliverable — we never
# clarify on it (the user wants the file, not more questions).
_DOWNLOAD_RE = _re_attach.compile(
    r"\b(zip|\.zip|archive|compressed?)\b", _re_attach.IGNORECASE,
)


def _is_download_request(text: str) -> bool:
    t = (text or "").lower()
    if _DOWNLOAD_RE.search(t):
        return True
    return "download" in t and bool(_re_attach.search(
        r"\b(project|projects|code|codebase|app|application|source|files?|"
        r"everything|repo|repository)\b", t))





async def _race_clarifier(agen, clarify_task):
    """Race the answer's first token (from async generator `agen`) against the
    Clarifier task (which returns a question list). Option-B semantics:
      • returns (None, questions) when the clarifier produces questions BEFORE
        the first token (or within the grace window) → caller emits a clarify
        event and drops the answer;
      • returns (first_chunk, None) when the answer starts first (or the
        clarifier declines) → caller streams; the clarifier is cancelled.
    Near-zero added latency to the answer: a declining clarifier just stops
    racing, and a fast answer waits at most `clarify_grace_ms` for the gate."""
    anext_task = asyncio.ensure_future(agen.__anext__())
    pending = {anext_task}
    if clarify_task is not None:
        pending.add(clarify_task)
    while pending:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED)
        if clarify_task is not None and clarify_task in done:
            try:
                qs = clarify_task.result() or []
            except Exception:  # noqa: BLE001
                qs = []
            if qs:
                anext_task.cancel()
                with contextlib.suppress(BaseException):
                    await anext_task
                return None, qs
            clarify_task = None  # declined — keep waiting for the first token
            continue
        if anext_task in done:
            if clarify_task is not None:
                qs = await _await_clarifier_grace_task(clarify_task)
                if qs:
                    anext_task.cancel()
                    with contextlib.suppress(BaseException):
                        await anext_task
                    return None, qs
            try:
                return anext_task.result(), None
            except StopAsyncIteration:
                return None, None
    return None, None


def _clarify_grace_s() -> float:
    """Grace window (seconds) for the clarifier to interrupt a fast answer."""
    try:
        from app.core.config_loader import cfg
        return max(0.0, int(cfg.advanced_rag.clarify_grace_ms) / 1000.0)
    except Exception:  # noqa: BLE001
        return 1.5


async def _await_clarifier_grace_task(clarify_task) -> list:
    """Give the clarifier a brief grace window to interrupt a fast answer.
    Returns its question list (the task's result) or [] on timeout/error."""
    grace = _clarify_grace_s()
    if grace <= 0:
        clarify_task.cancel()
        return []
    try:
        await asyncio.wait_for(asyncio.shield(clarify_task), timeout=grace)
    except asyncio.TimeoutError:
        clarify_task.cancel()
        return []
    except Exception:  # noqa: BLE001
        return []
    try:
        return clarify_task.result() or []
    except Exception:  # noqa: BLE001
        return []


@router.get("/chat/attachment-image")
async def get_chat_attachment_image(path: str):
    """Serve a persisted chat image (from sources['images'][].path) so the
    client can re-attach it on retry / after a reload."""
    from fastapi.responses import Response

    if not path.startswith("chat_images/") or ".." in path:
        raise HTTPException(404, detail="Not found")
    try:
        from storage.blobs import get_blobs

        data = await get_blobs().get(path)
    except Exception:  # noqa: BLE001
        raise HTTPException(404, detail="Image not found")
    name = path.rsplit("/", 1)[-1].lower()
    mime = (
        "image/png" if name.endswith(".png")
        else "image/webp" if name.endswith(".webp")
        else "image/gif" if name.endswith(".gif")
        else "image/jpeg"
    )
    return Response(content=data, media_type=mime)


# Strong refs to detached background tasks (partial-save on disconnect,
# auto-title) so the loop doesn't GC them before they finish.
_BG_SAVES: set = set()

# Inline the full document text when the combined size is under this (chars).
# Max document text fed inline to the model. ~120k chars ≈ 30k tokens — safe
# for most models' context windows and fast to send. Larger docs are sampled
# (head + tail) so the reply stays fast; the FULL doc is embedded in the
# background for follow-up RAG (the Retriever agent queries it next turn).
_INLINE_BUDGET_CHARS = 120_000
# Archive extensions we try to turn into a code knowledge graph.
_ARCHIVE_EXTS = (".zip", ".7z", ".tar", ".tgz", ".tbz2", ".txz", ".tzst",
                 ".tar.gz", ".tar.bz2", ".tar.xz")


async def _read_capped(uf: UploadFile, limit: int) -> bytes | None:
    """Read an upload in 1 MB chunks, bailing out if it exceeds `limit`.

    Returns the bytes, or None if the file is over the limit — so a multi-GB
    upload is never fully materialized in memory just to be rejected. Caps peak
    memory at ~limit + 1 MB per file.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await uf.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _human_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB"


def _sse(event: str, data: dict) -> str:
    def _default(o):
        import uuid as _uuid
        from datetime import datetime as _dt

        if isinstance(o, _uuid.UUID):
            return str(o)
        if isinstance(o, _dt):
            return o.isoformat()
        raise TypeError(f"{type(o).__name__} not serializable")

    return f"event: {event}\ndata: {json.dumps(data, default=_default)}\n\n"


def _build_doc_context(docs: list[tuple[str, str]]) -> str:
    """Build this turn's inline document context — FAST (no embedding).

    Small docs are inlined whole; oversized docs are sampled (head + tail) so
    the reply starts immediately instead of waiting on RAG. Full documents are
    embedded in the background (see the upload route) for follow-up retrieval.
    """
    total = sum(len(t) for _, t in docs)
    if total <= _INLINE_BUDGET_CHARS:
        return "\n\n".join(f"--- Document: {fn} ---\n{t}" for fn, t in docs)

    per_doc = max(8_000, _INLINE_BUDGET_CHARS // max(1, len(docs)))
    pieces: list[str] = []
    for fn, t in docs:
        if len(t) <= per_doc:
            pieces.append(f"--- Document: {fn} ---\n{t}")
            continue
        head = t[: (per_doc * 7) // 10]
        tail = t[-(per_doc * 3) // 10:]
        pieces.append(
            f"--- Document: {fn} ({len(t):,} chars — showing the first and "
            f"last parts; middle omitted) ---\n{head}\n\n"
            f"…[middle of {fn} omitted]…\n\n{tail}"
        )
    return "\n\n".join(pieces)


@router.post("/chat/upload-stream")
async def chat_upload_stream(
    message: str = Form(...),
    conversation_id: str | None = Form(None),
    session_id: str | None = Form(None),  # noqa: ARG001 — reserved for parity
    depth: str | None = Form(None),
    instruction: str | None = Form(None),  # hidden model directive (not saved)
    difficulty: str | None = Form(None),  # manual effort override; None = auto
    passwords: str = Form("{}"),  # JSON {filename: password} for protected files
    files: list[UploadFile] = File(default=[]),
    session: AsyncSession = Depends(get_session),
):
    """Stream an answer over the message + uploaded documents/images.

    `message` is what the user sees/saves in their bubble; `instruction` (when
    set) is a hidden directive the MODEL receives instead — e.g. the Solve
    action shows "Solve this." but instructs the model to solve the screenshot.
    """
    from storage import bootstrap as _bs

    if not _bs.POSTGRES_READY:
        raise HTTPException(503, detail="Database not ready. Open Settings → Database.")
    if not message.strip() and not files:
        raise HTTPException(400, detail="Provide a message or at least one file.")

    # What the MODEL is asked — the hidden `instruction` when present, else the
    # user's message. The saved user_msg (the bubble) always keeps `message`.
    model_message = (instruction.strip()
                     if (instruction and instruction.strip()) else message)

    # Passwords the FE supplies for protected files (filename -> password).
    try:
        pw_map: dict = json.loads(passwords or "{}")
        if not isinstance(pw_map, dict):
            pw_map = {}
    except Exception:  # noqa: BLE001
        pw_map = {}

    # --- Parse attachments: images → base64; documents → text. ---
    images: list[str] = []
    image_refs: list[dict] = []  # persisted so retry/reload can re-attach
    file_refs: list[dict] = []   # documents persisted for preview + instant reload
    docs: list[tuple[str, str]] = []
    archives: list[tuple[str, bytes]] = []   # (filename, raw bytes) for code graph
    attachment_names: list[str] = []
    parse_errors: list[str] = []
    needs_password: list[str] = []  # protected files awaiting a password
    # Content-hash → already-stored blob path, scoped to THIS request so the
    # same file (or the same pasted block) attached twice in one turn stores one
    # blob and both refs point at it. Request-scoped only: cross-conversation
    # sharing would be unsafe because delete_conversation removes a thread's
    # blobs (a shared blob would break the other conversation's preview).
    import hashlib

    _seen_doc: dict[str, str] = {}
    _seen_img: dict[str, str] = {}
    for uf in files:
        fn = uf.filename or "file"
        # Fast reject when the part advertises its size; otherwise the capped
        # read below enforces the limit without materializing the whole file.
        if uf.size is not None and uf.size > MAX_UPLOAD_BYTES:
            parse_errors.append(
                f"{fn}: {_human_mb(uf.size)} exceeds the "
                f"{_human_mb(MAX_UPLOAD_BYTES)} per-file limit — skipped."
            )
            continue
        data = await _read_capped(uf, MAX_UPLOAD_BYTES)
        if data is None:
            parse_errors.append(
                f"{fn}: exceeds the {_human_mb(MAX_UPLOAD_BYTES)} per-file "
                "limit — skipped."
            )
            continue
        if not data:
            continue
        attachment_names.append(fn)
        # Stash raw archive bytes so a project zip/tar can be turned into a code
        # knowledge graph below (in ADDITION to its text being extracted for RAG).
        if fn.lower().endswith(_ARCHIVE_EXTS):
            archives.append((fn, data))
        if is_image(fn):
            images.append(base64.b64encode(data).decode("ascii"))
            # Persist the image so a later retry (or after a reload) can
            # re-attach it — the DB row otherwise only keeps the filename.
            try:
                import uuid as _uuid

                from storage.blobs import get_blobs

                _h = hashlib.sha256(data).hexdigest()
                path = _seen_img.get(_h)
                if path is None:
                    path = f"chat_images/{_uuid.uuid4().hex}_{fn}"
                    await get_blobs().put(path, data)
                    _seen_img[_h] = path
                image_refs.append({"name": fn, "path": path})
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
            continue
        # Read the document/code/archive (with a password if the FE supplied
        # one). A protected file with no/wrong password is collected so the FE
        # can prompt the user, Claude-style, and retry.
        try:
            text = extract_document_text(data, fn, password=pw_map.get(fn))
        except PasswordRequired:
            needs_password.append(fn)
            continue
        except UnsupportedDocument as exc:
            parse_errors.append(str(exc))
            continue
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{fn}: could not read ({exc})")
            continue
        # Persist the raw file (readable only) so it can be opened in the
        # preview panel and reloaded instantly (stored in Postgres). ARCHIVES
        # are excluded — their bytes are used for the code graph, but a .zip/.7z
        # is not previewable, so we neither store a preview blob nor emit a
        # file_ref for one (the text extraction below still feeds RAG).
        if not fn.lower().endswith(_ARCHIVE_EXTS):
            try:
                import uuid as _uuid

                from storage.blobs import get_blobs

                _h = hashlib.sha256(data).hexdigest()
                dpath = _seen_doc.get(_h)
                if dpath is None:
                    dpath = f"documents/{_uuid.uuid4().hex}_{fn}"
                    await get_blobs().put(dpath, data)
                    _seen_doc[_h] = dpath
                file_refs.append({"name": fn, "path": dpath})
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
        if text.strip():
            docs.append((fn, text))

    # A protected file needs a password before we can read it — tell the FE to
    # prompt the user (Claude-style), then retry the upload with `passwords`.
    if needs_password:
        # The turn is aborted (the FE retries with passwords), so roll back any
        # blobs persisted this attempt — otherwise mixing a protected file with
        # readable ones would orphan the readable ones' blobs on every retry.
        try:
            from storage.blobs import get_blobs

            _store = get_blobs()
            for _ref in (*image_refs, *file_refs):
                try:
                    await _store.delete(_ref["path"])
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(
            status_code=423,
            content={"detail": "password_required", "files": needs_password},
        )

    # --- Conversation row. ---
    if conversation_id:
        convo = await session.get(Conversation, conversation_id)
        if convo is None:
            raise HTTPException(404, detail="Conversation not found")
    else:
        title = " ".join(message[:500].split()[:6])[:200] or "New conversation"
        convo = Conversation(title=title)
        session.add(convo)
        await session.flush()

    _src: dict = {}
    if attachment_names:
        _src["attachments"] = attachment_names
    if image_refs:
        _src["images"] = image_refs  # [{name, path}] → re-attachable on retry
    if file_refs:
        _src["files"] = file_refs    # [{name, path}] → previewable documents
    # Persist the hidden model directive (e.g. the Solve "write in the SELECTED
    # language" instruction) so a RETRY re-passes it — the bubble still shows
    # only `message`. Without this, a retried Solve dropped the directive and
    # regressed to the wrong language.
    if instruction and instruction.strip() and \
            instruction.strip() != (message or "").strip():
        _src["instruction"] = instruction.strip()[:4000]
    # §12: the turn's input modality (image | document | audio | multimodal),
    # carried into the assistant envelope + the `done` event. Fail-open to text.
    try:
        from app.response_arch.envelope import build_input as _build_input
        _input_block = _build_input(
            text=message, images=images, files=(docs + archives))
    except Exception:  # noqa: BLE001
        _input_block = {"modality": "text"}
    _input_modality = _input_block.get("modality", "text")
    user_msg = Message(
        conversation_id=convo.id,
        role="user",
        content=message,
        sources=_src or None,
    )
    session.add(user_msg)
    await session.flush()

    intent = plan(model_message or "Analyze the attached file(s).")

    # Prior turns (exclude the just-added user message — we re-add it below
    # carrying the images, which the DB row doesn't store).
    history = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == convo.id, Message.id != user_msg.id)
            .order_by(Message.created_at)
        )
    ).scalars().all()

    # Persist every uploaded document's vectors into the conversation's
    # collection (reused by follow-up turns via the Retriever agent), then
    # build this turn's context.
    cid = str(convo.id)
    if docs:
        # Embed the full documents in the BACKGROUND so a big file (e.g. a 2 MB
        # log) doesn't stall the reply on CPU embedding. Follow-up turns query
        # these vectors via the Retriever agent. The detached task is held in
        # _BG_SAVES so the loop doesn't GC it.
        from app.rag.documents import ingest_chat_document

        async def _ingest_all() -> None:
            for fn, text in docs:
                try:
                    await ingest_chat_document(cid, fn, text)
                except Exception as exc:  # noqa: BLE001 — never block on ingest
                    log.info("chat-doc ingest failed for %s: %s", fn, exc)

        _t = asyncio.create_task(_ingest_all())
        _BG_SAVES.add(_t)
        _t.add_done_callback(_BG_SAVES.discard)

    # Window prior turns to a token budget (long threads don't resend it all);
    # the trimmed older turns are covered by the session's rolling summary.
    from app.chat.history import window_messages

    # Prior image turns store their vision analysis in `sources.image_analysis`
    # (not in `content`, so the bubble stays "Solve this."). Feed that analysis
    # back to the model here so a follow-up — e.g. after we asked which language
    # — still has the problem from the earlier screenshot.
    all_prior = []
    for m in history:
        c = m.content
        analysis = None
        if isinstance(getattr(m, "sources", None), dict):
            analysis = m.sources.get("image_analysis")
        if m.role == "user" and analysis:
            c = f"{c}\n\n[Attached image content]:\n{analysis}"
        all_prior.append({"role": m.role, "content": c})
    # Threaded: window_messages condenses each message, which is CPU work that
    # could stall the event loop if an old turn stored a huge body.
    prior, dropped = await asyncio.to_thread(window_messages, all_prior)
    history_summary = (convo.summary or "").strip() if dropped > 0 else ""

    # Start the Clarifier NOW (as a task) so its LLM call overlaps the doc/graph
    # work below instead of adding its latency on top; awaited just before the
    # generator. Declines on most turns (greetings, clear asks).
    _clarify_task = None
    _clarify_priority_upload = False
    _suppress_clarify_upload = _is_download_request(message or "")
    try:
        from app.chat.difficulty import is_ambiguous_build_request
        _recent_u = " ".join(
            m["content"] for m in prior if m.get("role") == "user"
        )
        _clarify_priority_upload = is_ambiguous_build_request(
            message or "", _recent_u
        )
    except Exception:  # noqa: BLE001
        _clarify_priority_upload = False
    # Phase-2 clarification elimination: detect the uploaded project's stack
    # (manifests → language/framework/build tool) so the clarifier never asks
    # what the upload already answers, and later turns inherit the decision
    # via the goal ledger. Deterministic, bounded, fail-open — an empty
    # profile changes nothing.
    _stack_slots: dict = {}
    if archives:
        try:
            from app.codeintel.stack_profile import detect_stack_from_archive
            for _saf, _sad in archives:
                _sp = detect_stack_from_archive(_sad, _saf)
                for _sk, _sv in _sp.slots().items():
                    _stack_slots.setdefault(_sk, _sv)
            if _stack_slots:
                log.info("stack profile detected from upload: %s", _stack_slots)

                async def _persist_stack_slots() -> None:
                    """Best-effort: record detected slots as decided for this
                    conversation (never re-asked on later turns)."""
                    try:
                        from app.clarify import GoalLedger
                        from storage.db import get_session_factory as _gsf2
                        from storage.device import ensure_device_user
                        from storage.models import User
                        _f2 = _gsf2()
                        if _f2 is None:
                            return
                        _uid = await ensure_device_user()
                        if _uid is None:
                            return
                        async with _f2() as _s2:
                            _u2 = await _s2.get(User, _uid)
                            if _u2 is None:
                                return
                            _root = dict(_u2.preferences or {})
                            _led = GoalLedger(_root, str(convo.id))
                            for _sk, _sv in _stack_slots.items():
                                _led.record_choice(_sk, str(_sv))
                            _u2.preferences = _root
                            await _s2.commit()
                    except Exception:  # noqa: BLE001 — never break the turn
                        pass

                _t_stack = asyncio.create_task(_persist_stack_slots())
                _BG_SAVES.add(_t_stack)
                _t_stack.add_done_callback(_BG_SAVES.discard)
        except Exception:  # noqa: BLE001
            _stack_slots = {}

    try:
        from app.core.config_loader import cfg as _cfgq
        # For image turns the Clarifier is (re)started INSIDE the generator —
        # image-aware after vision analysis — so we don't start a wasted
        # message-only one here. A download/zip turn never clarifies.
        if (_cfgq.advanced_rag.upload_quality_checks
                and (message or "").strip() and not images
                and not _suppress_clarify_upload):
            from app.chat.quality import maybe_clarify
            _clarify_task = asyncio.create_task(
                maybe_clarify(message, prior,
                              clarify_priority=_clarify_priority_upload,
                              has_artifact=True,
                              attachment_slots=_stack_slots or None))
    except Exception:  # noqa: BLE001
        _clarify_task = None

    # Single-call turn triage (difficulty + document-intent) — one LLM round-
    # trip instead of two — overlapped with the work below. A code archive
    # raises the floor to a heavier task via the hint.
    _triage_task = None
    try:
        from app.core.config_loader import cfg as _cfgd
        if _cfgd.advanced_rag.difficulty_aware_routing:
            from app.chat.triage import triage as _triage
            _recent_up = " ".join(
                m["content"] for m in prior if m.get("role") == "user"
            )
            _triage_task = asyncio.create_task(_triage(
                (message or "")
                + (" [analyzing a code project]" if archives else ""),
                recent=_recent_up,
            ))
    except Exception:  # noqa: BLE001
        _triage_task = None

    doc_context = _build_doc_context(docs) if docs else ""

    # Code knowledge graph: turn any uploaded project archive into a symbol +
    # relationship graph (tree-sitter, like codegraph) and inject a project
    # overview so the model understands the codebase's STRUCTURE, not just its
    # raw text. Built in a worker thread; never blocks/breaks the turn.
    code_graph_block = ""
    if archives:
        try:
            from app.core.config_loader import cfg as _cfg
            if _cfg.advanced_rag.use_code_knowledge_graph:
                from app.codegraph.ingest import ingest_archive_code_graph

                # Build concurrently with a time budget: graphs that finish in
                # time enrich THIS answer; a slow build (huge repo) keeps running
                # in the background — its overview is available on the next turn
                # via the RAG-embedded graph, so the first token is never blocked.
                _tasks = [asyncio.create_task(
                    ingest_archive_code_graph(str(convo.id), _ad, _af))
                    for _af, _ad in archives]
                _done, _pending = await asyncio.wait(_tasks, timeout=10)
                for _t in _pending:
                    _BG_SAVES.add(_t)
                    _t.add_done_callback(_BG_SAVES.discard)
                _summaries = []
                for _t in _done:
                    try:
                        _r = _t.result()
                    except Exception:  # noqa: BLE001
                        _r = None
                    if _r:
                        _summaries.append(_r[0])
                if _summaries:
                    code_graph_block = (
                        "\n\n".join(_summaries)
                        + "\n\n----- (end of code knowledge graph) -----\n\n"
                    )
        except Exception as exc:  # noqa: BLE001 — never break chat on graph build
            log.info("code graph build failed: %s", exc)

    # Workspace materializer (Phase 2): extract uploaded code archive(s) into a
    # real, sandboxed, editable project folder keyed by conversation, so the
    # chat agent-run endpoint (Phase 3) can read / edit / build / verify the
    # actual codebase — not just its embedded text. Best-effort, time-budgeted,
    # flag-gated; failures NEVER block or break the turn.
    if archives:
        try:
            from app.core.config_loader import cfg as _cfgm
            if getattr(_cfgm.advanced_rag, "materialize_workspace", True):
                from app.agent_workspace import (
                    git_init_baseline,
                    materialize_archive,
                    workspace_path,
                )

                async def _materialize_one(_af: str, _ad: bytes) -> None:
                    res = await asyncio.to_thread(
                        materialize_archive, str(convo.id), _ad, _af)
                    if res.ok:
                        try:
                            await git_init_baseline(workspace_path(str(convo.id)))
                        except Exception:  # noqa: BLE001
                            pass
                        log.info("materialized workspace for %s: %d files, "
                                 "%d bytes%s", convo.id, res.files, res.bytes,
                                 " (truncated)" if res.truncated else "")

                # Materialize only the FIRST code archive (a conversation maps to
                # one workspace); extra archives are ignored for the workspace.
                _mat_tasks = [
                    asyncio.create_task(_materialize_one(_af, _ad))
                    for _af, _ad in archives[:1]
                ]
                _mdone, _mpending = await asyncio.wait(_mat_tasks, timeout=8)
                for _t in _mpending:
                    _BG_SAVES.add(_t)
                    _t.add_done_callback(_BG_SAVES.discard)
        except Exception as exc:  # noqa: BLE001 — never break chat on materialize
            log.info("workspace materialize failed: %s", exc)

    system_prompt = intent.system_prompt
    if history_summary:
        system_prompt += (
            "\n\nEarlier in this same conversation (summary of older turns no "
            "longer shown verbatim — established context the user may refer "
            "back to):\n" + history_summary
        )
    # Document CONTENT goes into the user turn (below), not the system prompt —
    # placing it adjacent to the question means recency weighting anchors the
    # answer on the NEW file. The system prompt only states the FACTS of this
    # turn's attachment(s); understanding the user's intent (greeting, vague
    # ask, specific question, gibberish, …) and responding appropriately is the
    # MODEL's job — no hardcoded keyword rules, works with any provider. Images
    # carry no text block (the model sees them via vision).
    doc_block = ""
    img_names = [n for n in attachment_names if is_image(n)]
    if docs or images:
        kinds: list[str] = []
        if docs:
            kinds.append("document(s)")
        if images:
            kinds.append("image(s)")
        noun = " and ".join(kinds)
        new_names = ", ".join([fn for fn, _ in docs] + img_names) or "the attachment"
        system_prompt += (
            f"\n\nThe user has attached the following {noun} in THIS message: "
            f"{new_names}. They were just attached and are the subject of this "
            "turn — do not confuse them with any file or image from earlier in "
            "the conversation; focus only on these new one(s)"
            + (" (the document text is included at the start of the user's "
               "message below)" if docs else "")
            + ".\n\nRead/inspect them and respond naturally to whatever the "
            "user's message actually is — judge their intent yourself:\n"
            "• if they greeted you or made small talk, warmly greet them back "
            "first;\n"
            "• if the message is empty, unclear, or not a real request, briefly "
            "and kindly say you couldn't make out a specific ask;\n"
            "• if they only refer vaguely to the attachment (e.g. \"what is "
            f"this?\"), proactively give a clear overview of each {noun} — what "
            "it is, its purpose, key contents / structure (for an image, what it "
            "shows), and anything notable — then ask what they'd like to do "
            "next;\n"
            "• if they asked something specific, just answer it.\n"
            "Cite each attachment by name."
        )
        if doc_context:
            # §11 trust boundary: uploaded document text is UNTRUSTED — frame it
            # as data, not instructions, so a poisoned file can't hijack the turn.
            from app.response_arch.trust import frame_untrusted
            doc_block = (
                f"[Attached this turn: {new_names}]\n\n"
                + frame_untrusted(doc_context, label="attached document")
                + "\n\n"
            )

    # Difficulty (from the triage task started above): add the rigor directive
    # for demanding turns; the label also steers routing toward the strongest
    # model (passed to the stream below).
    # A manual effort override ("think harder") from the composer wins over the
    # triage classifier — same as the non-upload chat path.
    _manual_difficulty = (difficulty or "").strip().lower() or None
    if _manual_difficulty not in (None, "standard", "easy", "hard", "expert"):
        _manual_difficulty = None
    _difficulty = "standard"
    if _triage_task is not None:
        try:
            _difficulty = (await _triage_task).difficulty
        except Exception:  # noqa: BLE001
            _difficulty = "standard"
    if _manual_difficulty is not None:
        _difficulty = _manual_difficulty
    if _difficulty != "standard":
        try:
            from app.chat.difficulty import rigor_directive
            system_prompt += rigor_directive(_difficulty)
        except Exception:  # noqa: BLE001
            pass

    llm_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in prior:
        llm_messages.append({"role": m["role"], "content": m["content"]})
    # Cap the new turn too: a huge pasted body is reduced to its key lines
    # (window_messages only condenses the PRIOR turns, not this fresh one).
    # Threaded — the salience pass is CPU work that can stall on a giant paste.
    from app.chat.condense import condense_oversized

    _msg = (await asyncio.to_thread(condense_oversized, model_message))[0] if model_message else model_message
    # Prepend this turn's code-graph overview + document content so they sit
    # right next to the user's question (recency) — keeps the model anchored on
    # the new upload and aware of the project's structure.
    user_text = code_graph_block + doc_block + (_msg or "Analyze the attached file(s).")
    user_turn: dict = {"role": "user", "content": user_text}
    if images:
        user_turn["images"] = images
    llm_messages.append(user_turn)

    await session.commit()

    conversation_id_out = convo.id
    intent_label = intent.label
    convo_title = convo.title or ""  # captured for auto-title (avoid lazy load)

    # Agentic routing hint (Phase 3): does this turn want the AGENT LOOP (build
    # an app from a spec / edit an uploaded codebase) rather than a plain
    # answer? Deterministic, zero-latency. Surfaced as an `agentic` SSE frame so
    # the chat client (Phase 4) can re-issue the turn to `/api/chat/agent-run`
    # (where the workspace — just materialized above — is read/edited/built).
    _agentic_hint: dict = {"agentic": False, "kind": None}
    try:
        from app.agent_workspace import workspace_exists as _ws_exists
        from app.documents.detect import detect_agentic_intent as _detect_ag
        _has_spec_doc = bool(docs) and not archives
        _ws_present = bool(archives) or _ws_exists(str(convo.id))
        _agentic_hint = _detect_ag(
            message or "", has_archive=bool(archives),
            has_spec_doc=_has_spec_doc, workspace_exists=_ws_present)
    except Exception:  # noqa: BLE001
        _agentic_hint = {"agentic": False, "kind": None}

    # Document-intent comes from the SAME triage task as difficulty (one call) —
    # awaited at save time, by which point it's resolved.

    # The Clarifier task (started above) is NOT awaited here — it's raced against
    # the answer's first token inside the generator (Option B), so the answer
    # never waits on it.

    async def event_generator() -> AsyncGenerator[str, None]:
        # Explicit Stop: the FE POSTs /conversations/{id}/cancel (its HTTP client
        # can't abort a request mid-flight). Clear any STALE flag from a prior
        # turn so this one isn't killed at birth, then poll it between steps.
        from app.api.replay import is_cancelled as _is_cancelled, clear_cancel
        _cid = str(conversation_id_out)
        clear_cancel(_cid)
        yield _sse("meta", {"conversation_id": conversation_id_out, "intent": intent_label})
        if _agentic_hint.get("agentic"):
            yield _sse("agentic", {
                "conversation_id": str(conversation_id_out),
                "kind": _agentic_hint.get("kind"),
                "task": message or "",
            })
            # Hand off to the AGENT LOOP: emit a short note, persist it, and STOP
            # the normal answer. The chat client starts the agent-run SSE next
            # (`/api/chat/agent-run`), which reads/edits/builds the workspace we
            # just materialized. Skipping the normal answer avoids a wasteful
            # duplicate LLM call / full code dump on an agentic turn.
            _kind = _agentic_hint.get("kind")
            _note = (
                "Got it — I've loaded your "
                + ("project into a workspace" if _kind == "edit"
                   else "spec and prepared a workspace")
                + " and I'm starting work on it now. You'll see each step below."
            )
            yield _sse("token", {"text": _note})
            _mid = None
            try:
                from storage.db import get_session_factory as _gsf
                _f = _gsf()
                if _f is not None:
                    async with _f() as _ws:
                        _m = Message(
                            conversation_id=conversation_id_out,
                            role="assistant", content=_note,
                            intent=intent_label,
                            sources={"agentic": True, "kind": _kind})
                        _ws.add(_m)
                        _crow = await _ws.get(Conversation, conversation_id_out)
                        if _crow is not None:
                            _crow.title = _crow.title  # bump updated_at
                        await _ws.commit()
                        await _ws.refresh(_m)
                        _mid = str(_m.id)
            except Exception as exc:  # noqa: BLE001
                log.warning("agentic handoff save failed: %s", exc)
            yield _sse("done", {"message_id": _mid, "agentic": True,
                                "kind": _kind})
            return
        log.info("upload-stream: %d image(s), %d doc(s), %d archive(s) — msg=%r",
                 len(images), len(docs), len(archives), (message or "")[:60])

        # ── Vision → task pipeline ───────────────────────────────────────
        # If the user attached image(s), a VISION model first reads them into a
        # text description; we then strip the image off the turn so the actual
        # task is run by a SECOND model routed by difficulty (a heavy task can
        # use a strong, possibly non-vision model). EXCEPTION: visually-exact
        # intents (clone/replicate/pixel-perfect) KEEP the image on the task
        # model — a text description would lose fidelity. On failure we fall
        # back to the single-model path (image stays on the turn).
        clarify_task = _clarify_task

        # Set by the image branch; read by the post-stream sandbox verifier.
        analysis = ""
        _img_lang = None
        _img_lang_label = None

        def _start_clarify(text_for_gate: str, extra_slots: dict | None = None):
            if _suppress_clarify_upload:
                return None  # download/zip turn → never clarify
            try:
                from app.core.config_loader import cfg as _cfgc
                if (_cfgc.advanced_rag.upload_quality_checks
                        and (message or "").strip()):
                    from app.chat.quality import maybe_clarify
                    # Merge the archive's stack slots with any per-call slots
                    # (e.g. the language read off an image) — both satisfy
                    # required slots so the gate never re-asks what the upload
                    # already answers.
                    _slots = dict(_stack_slots or {})
                    if extra_slots:
                        _slots.update({k: v for k, v in extra_slots.items() if v})
                    return asyncio.create_task(maybe_clarify(
                        text_for_gate, prior,
                        clarify_priority=_clarify_priority_upload,
                        # This route ALWAYS carries uploaded files/images —
                        # an artifact-missing ask here would be absurd.
                        has_artifact=True,
                        attachment_slots=_slots or None))
            except Exception:  # noqa: BLE001
                pass
            return None

        if images:
            from app.core.config_loader import cfg as _cfg_vis
            _local_vision = bool(getattr(_cfg_vis.vision, "enabled", True))
            visual_exact = bool(_VISUAL_EXACT_RE.search(message or ""))
            if visual_exact and not _local_vision:
                # LEGACY (local vision OFF): keep the image on the task model so
                # a pixel-exact ask isn't lossily described. With local vision ON
                # we parse even these locally — no raw image reaches a provider.
                clarify_task = _start_clarify(model_message or "")
            else:
                yield _sse("stage", {"name": "Reading image"})
                _extract_task = model_message or ""
                if visual_exact:
                    _extract_task += (
                        " Reproduce this exactly: capture every UI element, the "
                        "exact text, colours, spacing and layout precisely.")
                analysis = ""
                ocr_text = ""       # PURE local OCR (exact — never hallucinated)
                vlm_text = ""       # PURE vision-model description
                try:
                    # Keepalive during the (possibly >30s) vision read so the
                    # client's idle watchdog doesn't close the stream and leave
                    # an empty bubble. ONE vision read now yields all three parts
                    # — no separate/second OCR pass.
                    _vtask = asyncio.ensure_future(
                        _vision_extract(images, _extract_task))
                    async for _ka in _keepalive_until(_vtask):
                        yield _ka
                    analysis, ocr_text, vlm_text = _vtask.result()
                except Exception as exc:  # noqa: BLE001 — single-model fallback
                    log.info("vision extract failed (fallback): %s", exc)
                    analysis, ocr_text, vlm_text = "", "", ""
                # Auto-detect the programming language the user SELECTED in the
                # image so we solve in THAT language instead of asking. The key
                # rule: NEVER run detection on the merged `analysis` — it mixes
                # the exact OCR with the VLM's (possibly hallucinated) solution,
                # and a hallucinated `using System`/`Console.` would beat the real
                # Dart code. Detect on the OCR and VLM texts SEPARATELY, and pick
                # which to trust by the vision model's CAPABILITY.
                _img_lang = None        # canonical id (clarifier slot)
                _img_lang_label = None  # exact label incl. version (solver)
                _is_code_problem = False
                try:
                    from app.codeintel.code_language import (
                        _CANON_TO_LABEL,
                        detect_language,
                        detect_language_label,
                        looks_like_coding_problem,
                        requested_language,
                    )
                    from app.vision.factory import vision_capability
                    # SAFETY NET: the vision model read nothing but OCR did → use
                    # the OCR text as the analysis so the user still gets an
                    # answer (not "couldn't read the image").
                    if not (analysis or "").strip() and len(ocr_text.strip()) > 40:
                        analysis = ocr_text
                        log.info("vision: analysis empty — using OCR text "
                                 "(%d chars) as the extraction", len(ocr_text))
                    _is_code_problem = looks_like_coding_problem(
                        f"{analysis}\n{ocr_text}\n{message or ''}")
                    _capable = vision_capability() == "capable"
                    _ocr_lang = detect_language(ocr_text) if ocr_text else None
                    _vlm_lang = detect_language(vlm_text) if vlm_text else None
                    if _capable:
                        # TRUST the capable model (cloud / large local). OCR is a
                        # free cross-check — prefer it only when it DEFINITIVELY
                        # read a language and the model disagrees.
                        _img_lang = _vlm_lang or _ocr_lang
                        if _ocr_lang and _vlm_lang and _ocr_lang != _vlm_lang:
                            log.info("vision(capable): OCR cross-check %s → %s",
                                     _vlm_lang, _ocr_lang)
                            _img_lang = _ocr_lang
                    else:
                        # Small local model hallucinates → OCR is authoritative.
                        _img_lang = _ocr_lang or _vlm_lang
                    # Label from whichever text produced the winning language.
                    if _img_lang and _img_lang == _ocr_lang:
                        _img_lang_label = detect_language_label(ocr_text)
                    elif _img_lang and _img_lang == _vlm_lang:
                        _img_lang_label = detect_language_label(vlm_text)
                    # HIGHEST priority: the user EXPLICITLY named a language
                    # ("solve this in Swift"). A direct instruction OUTRANKS every
                    # image read. Scan BOTH the visible message AND the hidden
                    # `instruction` (the Solve button sends the bubble as "Solve
                    # this." but the language preference rides in the directive) —
                    # checking `message` alone missed a Swift/Dart/... request that
                    # was only ever in the instruction.
                    _req_text = "\n".join(
                        t for t in (message, instruction) if t and t.strip())
                    if _is_code_problem and _req_text:
                        _req = requested_language(_req_text)
                        if _req and _req != _img_lang:
                            log.info("image coding-problem: request named "
                                     "language %s → %s", _img_lang, _req)
                            _img_lang = _req
                            # Label MUST match the overridden language, never the
                            # stale image label (else lang=swift but label=Java).
                            _img_lang_label = (
                                detect_language_label(_req_text)
                                or _CANON_TO_LABEL.get(_req)
                                or _req.capitalize())
                    if not _img_lang and _is_code_problem:
                        from app.core.config_loader import cfg as _cfgL
                        # The language couldn't be read from the screenshot (a
                        # tiny editor chip in a full-desktop capture is easy to
                        # miss) and the user named none. Silently defaulting to
                        # Python is what produced "solved in Python when Swift was
                        # selected". Prefer to ASK: leave _img_lang UNRESOLVED so
                        # the clarifier + the solve directive's own "ask which
                        # language" branch fire, instead of forcing Python onto
                        # the solver. Opt back into the old behaviour via config.
                        if bool(getattr(_cfgL.code_solver,
                                        "ask_when_language_unknown", True)):
                            log.info("image coding-problem: language UNREAD and "
                                     "unrequested — will ASK (no silent default)")
                        else:
                            _img_lang = str(getattr(
                                _cfgL.code_solver, "default_language", "python"))
                            _img_lang_label = _img_lang_label or "Python3"
                    if _img_lang:
                        log.info("image coding-problem: language=%s (label=%s) "
                                 "[capable=%s ocr=%s vlm=%s]", _img_lang,
                                 _img_lang_label, _capable, _ocr_lang, _vlm_lang)
                except Exception:  # noqa: BLE001
                    _img_lang = None
                # Policy: with the local vision layer enabled, ALWAYS strip the
                # raw image so it never reaches a provider — even if extraction
                # came back empty (the model may still be downloading).
                if analysis or _local_vision:
                    _note = analysis or (
                        "[The attached image could not be read by the local "
                        "vision model — it may still be downloading. Check "
                        "Settings → Vision.]")
                    # Steer the solver to the detected language AND to build on
                    # the starter snippet shown in the image (LeetCode/HackerRank
                    # give a class/method stub you must fill in) — not invent a
                    # different structure. The full stub is already in the
                    # analysis above; this makes completing it explicit.
                    _lbl = _img_lang_label or _img_lang
                    _lang_directive = (
                        f"\n\nThe user is solving this coding problem in {_lbl}. "
                        f"Requirements:\n"
                        f"- Write the solution in EXACTLY {_lbl} — match the "
                        f"language AND its version/dialect precisely (e.g. "
                        f"Python3 and Python2 have different syntax; C++17 vs "
                        f"older). Use idiomatic {_lbl}.\n"
                        f"- COMPLETE the starter snippet shown in the analysis "
                        f"above: keep its exact class name, method name and "
                        f"signature; only fill in the body. If the snippet is "
                        f"not fully legible, use the standard platform signature "
                        f"for this problem in {_lbl}, not a standalone script.\n"
                        f"- Give the OPTIMAL solution: the best achievable time "
                        f"complexity and the best space complexity for this "
                        f"problem — never a brute force when a better bound "
                        f"exists. Handle every edge case.\n"
                        f"- State the Big-O time and space complexity and a "
                        f"one-line note on why it is optimal.\n"
                        f"Return the full, runnable solution."
                        if _img_lang else "")
                    for _m in reversed(llm_messages):
                        if _m.get("role") == "user":
                            _m.pop("images", None)
                            _m["content"] = (
                                _m["content"]
                                + "\n\n[Analysis of the attached image(s)]:\n"
                                + _note
                                + _lang_directive
                            )
                            break
                if analysis:
                    # Persist the analysis on the user turn (in `sources`, so
                    # the bubble text stays clean) so a follow-up — e.g. after
                    # we ask which language — still has the problem context.
                    try:
                        from storage.db import get_session_factory as _gsf
                        _f = _gsf()
                        if _f is not None:
                            async with _f() as _uws:
                                _urow = await _uws.get(Message, user_msg.id)
                                if _urow is not None:
                                    _src2 = dict(_urow.sources or {})
                                    _src2["image_analysis"] = analysis[:8000]
                                    _urow.sources = _src2
                                    await _uws.commit()
                    except Exception:  # noqa: BLE001
                        pass
                    # Clarifier sees the image content → image-grounded asks.
                    # A language read off the image is passed as a known slot so
                    # the gate never asks "which language?" for a problem that
                    # already shows one.
                    clarify_task = _start_clarify(
                        (model_message or "")
                        + "\n\n[Attached image content]:\n" + analysis,
                        extra_slots={"language": _img_lang} if _img_lang else None,
                    )
                else:
                    # No usable text (local model still downloading, or the
                    # legacy remote path failed with local vision off) → clarify
                    # on the message text. The image was already stripped above
                    # when local vision is the policy.
                    clarify_task = _start_clarify(
                        model_message or "",
                        extra_slots={"language": _img_lang} if _img_lang else None,
                    )

        from app.response_arch.sanitize import strip_reasoning
        from storage.db import get_session_factory
        from storage.repos import SessionRepo

        # Holds the authoritative document decision computed in _save, so the
        # `done` event can carry it (upload turns skip the post-stream reload,
        # so the client must learn the decision from `done`, not from sources).
        _saved_doc: dict = {"v": None}

        # Save the (possibly partial) assistant turn in a fresh session — safe
        # under cancellation. `bump` records the message counts on the normal
        # path; the disconnect/partial path skips it (best-effort).
        async def _save(text: str, *, incomplete: bool, bump: bool):
            f = get_session_factory()
            if f is None or not text.strip():
                return None
            async with f() as ws:
                _doc = None
                if _triage_task is not None:
                    try:
                        _tri = await _triage_task
                        # Don't let a recent "make a zip/doc" turn bleed into an
                        # image-analysis turn: on image turns, only flag a
                        # document when the CURRENT message actually asks for a
                        # downloadable file.
                        _allow_doc = (not images) or _is_download_request(message)
                        if _tri.wants_document and _allow_doc:
                            _doc = {"document": True, "format": _tri.doc_format}
                    except Exception:  # noqa: BLE001
                        _doc = None
                _saved_doc["v"] = _doc
                # §12: persist a response.v1 envelope carrying the input modality
                # so a reload reconstructs the same multimodal descriptor it
                # streamed live. Additive + fail-open.
                _env = None
                try:
                    from app.response_arch.envelope import build_envelope as _be
                    _env = _be(
                        conversation_id=str(conversation_id_out),
                        intent={"type": intent_label} if intent_label else None,
                        incomplete=bool(incomplete),
                        document=_doc,
                        input=_input_block,
                        input_modality=_input_modality,
                    )
                except Exception:  # noqa: BLE001 — envelope is additive
                    _env = None
                msg = Message(
                    conversation_id=conversation_id_out,
                    role="assistant",
                    content=text,
                    intent=intent_label,
                    incomplete=incomplete,
                    sources=_doc,
                    envelope=_env,
                )
                ws.add(msg)
                if bump:
                    repo = SessionRepo(ws)
                    await repo.record_message(conversation_id_out)
                    await repo.record_message(conversation_id_out)
                convo_row = await ws.get(Conversation, conversation_id_out)
                if convo_row is not None:
                    convo_row.title = convo_row.title  # trigger onupdate
                await ws.commit()
                await ws.refresh(msg)
                return msg.id

        collected: list[str] = []
        stream_completed = False

        async def _answer_stream(fresh: bool = False):
            # Stream the model directly for a FAST first token. Self-refine
            # (draft→verify→revise) blocks the WHOLE reply before any token
            # shows — on an image/Solve turn that feels very slow — so image
            # turns always stream directly; text turns use self-refine only when
            # it applies (expert/hard per config). `fresh` = a retry after a
            # provider drop: use a distinct session key so sticky routing can't
            # send us straight back to the model that just failed.
            _verified = None
            if not images:
                try:
                    from app.chat.verify import verified_answer
                    _verified = await verified_answer(
                        llm_messages, difficulty=_difficulty)
                except Exception:  # noqa: BLE001
                    _verified = None
            if _verified is not None:
                from app.chat.verify import chunk_text
                for piece in chunk_text(_verified):
                    yield piece
            else:
                _sk = (f"{conversation_id_out}:retry" if fresh
                       else str(conversation_id_out))
                async for chunk in llm.stream_chat(
                    llm_messages, session_key=_sk,
                    # "Always finishes": a long coding-solve answer that hits the
                    # output-token ceiling or drops mid-stream is re-prompted from
                    # the partial and stitched seamlessly, instead of leaving the
                    # user a half answer with a "Response interrupted" bar. Live
                    # keeps this off (config default) to avoid re-phrased echoes.
                    options={"difficulty": _difficulty,
                             "mid_stream_continuation": True},
                ):
                    yield chunk

        # Verify-BEFORE-reveal: for a coding-problem screenshot we DON'T stream the
        # half-baked solution token-by-token. Instead we buffer it, run the whole
        # sandbox pipeline (write test program → compile → run → test the cases),
        # showing live progress steps the entire time, and reveal the COMPLETE,
        # already-verified answer at the end. Non-coding turns stream as before.
        _verify_first = bool(
            images and _img_lang and _is_code_problem and (analysis or "").strip())
        log.info("UPLOAD-DIAG vision: analysis=%dch ocr=%dch vlm=%dch "
                 "img_lang=%s code_problem=%s verify_first=%s difficulty=%s",
                 len(analysis or ""), len(ocr_text or ""), len(vlm_text or ""),
                 _img_lang, _is_code_problem, _verify_first, _difficulty)

        # Option B: race the first answer token against the Clarifier so a normal
        # answer never waits on it — only interrupt to ask if it decides FIRST.
        agen = _answer_stream()
        # Detached sub-tasks that must NOT outlive this turn. When the client
        # presses Stop / disconnects, Starlette cancels this generator — but a
        # bare `asyncio.ensure_future` keeps running (the LLM answer stream, the
        # sandbox verify+repair). Track them so the `finally` can cancel them,
        # otherwise "Stop" doesn't actually stop the backend work.
        _race = None
        _vtk = None
        try:
            try:
                # Emit SSE keepalive comments while waiting for the first token —
                # an expert turn's multi-round self-refine on a big model can be
                # silent for a minute+, and an idle HTTP response gets dropped
                # ("connection closed while receiving data"). Keepalives are
                # ignored by the client parser.
                _race = asyncio.ensure_future(
                    _race_clarifier(agen, clarify_task))  # noqa: F841 — cancelled in finally
                _ka_tick = 0
                while True:
                    _done, _ = await asyncio.wait({_race}, timeout=0.6)
                    if _done:
                        break
                    # Stop pressed before the first token → tear down now (poll
                    # ~0.6s so Stop is near-instant); keepalive only ~every 6s.
                    if _is_cancelled(_cid):
                        raise asyncio.CancelledError()
                    _ka_tick += 1
                    if _ka_tick % 10 == 0:
                        yield ": keepalive\n\n"
                first_chunk, questions = _race.result()
            except (LLMError, Exception) as exc:  # noqa: BLE001
                # Provider error before any token — nothing to save.
                yield _sse("error", {"detail": str(exc)})
                return
            if questions:
                with contextlib.suppress(BaseException):
                    await agen.aclose()
                yield _sse("clarify", {"questions": questions})
                return
            for err in parse_errors:
                if not _verify_first:
                    yield _sse("token", {"text": f"> ⚠ {err}\n\n"})
            try:
                if _verify_first:
                    yield _sse("stage", {"name": f"Solving in {_img_lang_label or _img_lang}"})
                if first_chunk is not None:
                    collected.append(first_chunk)
                    if not _verify_first:
                        yield _sse("token", {"text": first_chunk})
                async for chunk in agen:
                    # Stop pressed → abort generation immediately (don't wait for
                    # the whole answer to buffer). CancelledError unwinds to the
                    # finally, which closes `agen` + saves the partial.
                    if _is_cancelled(_cid):
                        raise asyncio.CancelledError()
                    collected.append(chunk)
                    if _verify_first:
                        # Buffer silently; a keepalive keeps the SSE alive while
                        # the "Solving…" step stays on screen (content still empty
                        # on the client, so the progress stepper shows).
                        yield ": keepalive\n\n"
                    else:
                        yield _sse("token", {"text": chunk})
                stream_completed = True
            except (LLMError, Exception) as exc:  # noqa: BLE001
                # Mid-stream provider drop (e.g. "Provider error: NVIDIA NIM").
                # If it died EARLY, auto-retry with a fresh route (the circuit
                # breaker steers to a healthier provider) so a transient hiccup
                # doesn't leave the user with no answer. A `reset` clears the
                # false-start text on the client first. A late drop keeps the
                # near-complete partial rather than regenerating it.
                _early = len("".join(collected).strip()) < 600
                if _early and not stream_completed:
                    for _attempt in range(2):
                        collected.clear()
                        if not _verify_first:
                            yield _sse("reset", {})
                        try:
                            async for chunk in _answer_stream(fresh=True):
                                collected.append(chunk)
                                if _verify_first:
                                    yield ": keepalive\n\n"
                                else:
                                    yield _sse("token", {"text": chunk})
                            stream_completed = True
                            break
                        except (LLMError, Exception) as exc2:  # noqa: BLE001
                            exc = exc2
                            continue
                if not stream_completed:
                    partial = strip_reasoning("".join(collected)).strip()
                    if partial:
                        try:
                            await _save(partial, incomplete=True, bump=False)
                        except Exception:  # noqa: BLE001
                            pass
                    yield _sse("error", {"detail": str(exc)})
                    return

            full_text = strip_reasoning("".join(collected)).strip()
            log.info("UPLOAD-DIAG answer: first_pass=%dch chunks=%d "
                     "stream_completed=%s", len(full_text), len(collected),
                     stream_completed)
            if not full_text:
                # Empty (no error, just no content) — a flaky free model can
                # return nothing (or only hidden reasoning). Never leave a blank
                # turn: (1) retry the streaming pipeline on a FRESH route twice,
                # then (2) fall back to a DIRECT answer on ESCALATING tiers
                # (standard→hard→expert = different model pools) using the full
                # prompt. Only error if EVERY model returns nothing.
                for _eattempt in range(2):
                    collected.clear()
                    if not _verify_first:
                        yield _sse("reset", {})
                    try:
                        async for chunk in _answer_stream(fresh=True):
                            if _is_cancelled(_cid):
                                raise asyncio.CancelledError()
                            collected.append(chunk)
                            if _verify_first:
                                yield ": keepalive\n\n"
                            else:
                                yield _sse("token", {"text": chunk})
                    except asyncio.CancelledError:
                        raise
                    except (LLMError, Exception):  # noqa: BLE001 — try once more
                        continue
                    full_text = strip_reasoning("".join(collected)).strip()
                    if full_text:
                        break
                _last_fb_err = ""
                if not full_text:
                    from app.core.llm_client import llm as _fllm
                    for _diff in ("standard", "hard", "expert"):
                        if _is_cancelled(_cid):
                            raise asyncio.CancelledError()
                        try:
                            _txt, _ = await _fllm.complete_routed(
                                llm_messages, None, {"difficulty": _diff})
                        except Exception as _fbx:  # noqa: BLE001 — next tier
                            _last_fb_err = f"{type(_fbx).__name__}: {_fbx}"
                            log.info("UPLOAD-DIAG fallback tier=%s raised: %s",
                                     _diff, _last_fb_err)
                            continue
                        _txt = strip_reasoning(_txt or "").strip()
                        if _txt:
                            collected = [_txt]
                            full_text = _txt
                            if not _verify_first:
                                for _ci in range(0, len(_txt), 240):
                                    yield _sse("token",
                                               {"text": _txt[_ci:_ci + 240]})
                            log.info("upload: empty-answer recovered on tier=%s",
                                     _diff)
                            break
                        _last_fb_err = f"tier {_diff} returned empty"
                if not full_text:
                    # Surface the ACTUAL failure so it's diagnosable from the app,
                    # not a generic "empty response". An "Event loop is closed"
                    # here means the backend is running STALE code (restart it);
                    # a rate-limit/route error means the free models are exhausted.
                    _why = _last_fb_err or "every model returned nothing"
                    log.warning("UPLOAD-DIAG: answer EMPTY after all retries + "
                                "fallback — %s", _why)
                    # Put the reason IN the bubble (a visible token) so it's never
                    # a silent blank even on an un-rebuilt client, then the error.
                    _msg = ("⚠️ Couldn't get an answer — every model came back "
                            f"empty.\n\n**Reason:** {_why}\n\n"
                            "• If the reason mentions *Event loop is closed*, "
                            "fully **restart the backend** (it's running stale "
                            "code).\n• Otherwise the free models are momentarily "
                            "rate-limited/exhausted — **try again** or add a "
                            "reliable model in **Providers**.")
                    yield _sse("token", {"text": _msg})
                    full_text = _msg
                    msg_id = await _save(full_text, incomplete=True, bump=True)
                    yield _sse("done", {"message_id": msg_id,
                                        "user_message_id": str(user_msg.id)})
                    return
                log.info("UPLOAD-DIAG answer recovered: %dch", len(full_text))

            # Sandbox verification: for an image coding problem with a detected
            # language, COMPILE + RUN the solution against the visible examples
            # and append the verdict (✅ passed N/N / ⚠️ failed / ℹ️ no runtime).
            # The verdict is streamed AND folded into the saved answer.
            if images and _img_lang and (analysis or "").strip():
                try:
                    from app.codeintel.solution_verify import (
                        verify_and_maybe_repair,
                    )
                    _lbl = _img_lang_label or _img_lang
                    yield _sse("stage", {"name": "Reading the examples"})
                    # Stream REAL progress: verify_and_maybe_repair pings `on_stage`
                    # at each step (build harness → run in sandbox → check → maybe
                    # repair). We drain those onto the SSE stream so the turn shows
                    # live progress instead of sitting on "Waiting for reply…", and
                    # cap the whole thing with a deadline so `done` ALWAYS fires —
                    # a slow/absent LLM route at verify time can never wedge the turn.
                    _stage_q: asyncio.Queue = asyncio.Queue()

                    async def _on_stage(name: str):
                        await _stage_q.put(name)

                    # Tag every sandbox exec this verify spawns with the
                    # conversation id so Stop can KILL the running docker exec
                    # (not the container). The contextvar is captured by
                    # ensure_future and propagated into run_code's worker thread.
                    from app.sandbox import docker_exec as _dex
                    _dex.run_group.set(_cid)
                    _vtk = asyncio.ensure_future(
                        verify_and_maybe_repair(
                            analysis, full_text, _lbl, on_stage=_on_stage,
                            max_repairs=2,
                            # Seed the harness build at the tier the ANSWER used —
                            # that model pool just succeeded, so verification stops
                            # failing on rate-limited free "standard" models.
                            min_difficulty=_difficulty,
                        ))  # cancelled in the turn's finally on Stop
                    import time as _time
                    _deadline = _time.monotonic() + 180.0
                    _timed_out = False
                    _vka_tick = 0
                    while not _vtk.done():
                        # Stop pressed mid-verify → cancel the sandbox/repair task
                        # and unwind now (the finally releases everything). Poll
                        # ~0.6s so Stop is near-instant even during a silent
                        # compile; a real stage frame still flushes immediately.
                        if _is_cancelled(_cid):
                            _vtk.cancel()
                            # Kill the in-flight sandbox exec NOW (container stays
                            # up) so the compile/run stops with the turn.
                            try:
                                _dex.cancel_group(_cid)
                            except Exception:  # noqa: BLE001
                                pass
                            raise asyncio.CancelledError()
                        _remaining = _deadline - _time.monotonic()
                        if _remaining <= 0:
                            _vtk.cancel()
                            try:
                                _dex.cancel_group(_cid)
                            except Exception:  # noqa: BLE001
                                pass
                            _timed_out = True
                            break
                        try:
                            _name = await asyncio.wait_for(
                                _stage_q.get(), timeout=min(0.6, _remaining))
                            yield _sse("stage", {"name": _name})
                        except asyncio.TimeoutError:
                            _vka_tick += 1
                            if _vka_tick % 10 == 0:
                                yield ": keepalive\n\n"
                    while not _stage_q.empty():
                        yield _sse("stage", {"name": _stage_q.get_nowait()})
                    _suffix, _fixed = "", None
                    if _vtk.done() and not _vtk.cancelled():
                        try:
                            _suffix, _fixed = _vtk.result()
                        except Exception:  # noqa: BLE001
                            _suffix, _fixed = "", None
                    # ALWAYS attach a status line — the user must never see a
                    # coding answer with no verdict at all. Empty = timed out /
                    # errored before producing one.
                    if not _suffix:
                        _suffix = ("\n\n---\nℹ️ Sandbox verification timed out — "
                                   "the solution above is unchanged."
                                   if _timed_out else
                                   "\n\n---\nℹ️ Could not sandbox-verify this one "
                                   "— the solution above is unchanged.")
                    from app.codeintel.solution_verify import (
                        swap_code_block as _swap, _fence_tag as _ftag)
                    if _verify_first:
                        # Buffered: drop the VERIFIED corrected code in place of
                        # the buggy one so the reveal is one clean, working
                        # program (not broken code + a fix). Revealed below.
                        if _fixed:
                            full_text = _swap(full_text, _fixed, _lbl)
                        full_text = (full_text + _suffix).strip()
                    else:
                        # Streamed: the (possibly buggy) answer is already on
                        # screen — append the fix block (if any) + the verdict.
                        _delta = ((f"\n\n```{_ftag(_lbl)}\n{_fixed}\n```"
                                   if _fixed else "") + _suffix)
                        full_text = (full_text + _delta).strip()
                        yield _sse("token", {"text": _delta})
                except Exception:  # noqa: BLE001 — verification never breaks a turn
                    pass

                # Verify-before-reveal payoff: the answer was buffered (never
                # streamed), so now that the sandbox pipeline is done, reveal the
                # COMPLETE, verified answer at once — the "here we go" moment.
                if _verify_first:
                    yield _sse("stage", {"name": "Here we go"})
                    yield _sse("token", {"text": full_text})
            log.info("UPLOAD-DIAG reveal: %dch verify_first=%s", len(full_text),
                     _verify_first)

            msg_id = await _save(full_text, incomplete=False, bump=True)
            if msg_id is None:
                yield _sse("error", {"detail": "Database not bootstrapped."})
                return

            # `done` FIRST so the answer is never delayed. The grounding check is
            # an LLM call, so it runs AFTER done (the SSE stream stays open) and
            # flags any unsupported claims as a 'grounder flagged' tool chip —
            # same shape the mesh emits.
            yield _sse("done", {
                "message_id": msg_id,
                # The persisted USER-message id, so the client can reconcile its
                # optimistic `local-user-*` bubble to the real row. Without this
                # an edit/retry of an upload turn can't cascade-delete on the
                # server (the id check fails) → duplicated history on reload.
                "user_message_id": str(user_msg.id),
                # Authoritative doc decision (absent → not a document turn) — the
                # client opens a document strictly from this, matching the main
                # chat path. Upload turns skip the post-stream reload, so this is
                # the only live signal.
                **({"document": _saved_doc["v"]} if _saved_doc["v"] else {}),
                # §12: the input modality of this turn (image/document/multimodal).
                **({"modality": _input_modality}
                   if _input_modality and _input_modality != "text" else {}),
            })
            try:
                from app.core.config_loader import cfg as _cfgg
                if _cfgg.advanced_rag.upload_quality_checks and docs:
                    from app.chat.quality import check_grounding
                    # Bounded: this post-`done` LLM call must never hang the
                    # still-open SSE stream (a stuck/rate-limited free model
                    # would otherwise leave the connection open and the UI
                    # unable to start the next turn). On timeout we skip the
                    # grounding chip — the answer is already delivered.
                    _unv = await asyncio.wait_for(
                        check_grounding(full_text, docs), timeout=12.0)
                    if _unv:
                        # Live chip (mesh-style) AND persist on the message, so
                        # it's reliable regardless of post-`done` SSE timing and
                        # survives a reload.
                        yield _sse("tool", {"name": "grounder",
                                            "status": "flagged", "claims": _unv[:5]})
                        _f = get_session_factory()
                        if _f is not None:
                            async with _f() as _ws:
                                _row = await _ws.get(Message, msg_id)
                                if _row is not None:
                                    _src = dict(_row.sources or {})
                                    _src["grounding"] = _unv[:5]
                                    _row.sources = _src
                                    await _ws.commit()
            except Exception:  # noqa: BLE001
                pass

            from app.chat.auto_title import maybe_title

            t = asyncio.create_task(
                maybe_title(
                    conversation_id_out,
                    current_title=convo_title,
                    first_user=message,
                    first_assistant=full_text,
                )
            )
            _BG_SAVES.add(t)
            t.add_done_callback(_BG_SAVES.discard)

            from app.chat.history import maybe_update_summary

            st = asyncio.create_task(maybe_update_summary(conversation_id_out))
            _BG_SAVES.add(st)
            st.add_done_callback(_BG_SAVES.discard)
        except asyncio.CancelledError:
            # Client disconnected / Stop — save the partial in a detached task.
            if not stream_completed and collected:
                partial = strip_reasoning("".join(collected)).strip()
                if partial:
                    t = asyncio.create_task(_save(partial, incomplete=True, bump=False))
                    _BG_SAVES.add(t)
                    t.add_done_callback(_BG_SAVES.discard)
            raise
        finally:
            # Stop actually stops: cancel any detached sub-task still running and
            # close the LLM answer stream so the upstream provider request is
            # released. All no-ops on the normal (completed) path — the tasks are
            # already done and `agen` is exhausted. Never raises.
            for _pt in (_race, _vtk):
                if _pt is not None and not _pt.done():
                    _pt.cancel()
            # If a sandbox exec was mid-flight for this turn, kill it (container
            # stays up) — a disconnect/Stop that unwinds here must not leave the
            # compile/run churning in its worker thread.
            if _vtk is not None and not _vtk.done():
                with contextlib.suppress(BaseException):
                    from app.sandbox import docker_exec as _dex2
                    _dex2.cancel_group(_cid)
            with contextlib.suppress(BaseException):
                await agen.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
