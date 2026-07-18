"""Iterate the SOURCE files of an uploaded archive as (path, text) pairs.

Unlike the document parser (which concatenates everything into one text blob),
this preserves per-file structure — required to build a graph. Only files whose
extension maps to a supported language are yielded; binaries/oversized members
are skipped. Bomb guards (member count + per-file size) bound the work.
"""
from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from collections.abc import Iterator

from .tsutil import language_for

log = logging.getLogger(__name__)

_MAX_MEMBERS = 20_000
_MAX_FILE_BYTES = 1_500_000


def _decode(data: bytes) -> str | None:
    if not data or b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", "replace")


def _wanted(name: str, size: int) -> bool:
    return (language_for(name) is not None
            and 0 < size <= _MAX_FILE_BYTES)


def iter_source_members(data: bytes, filename: str) -> Iterator[tuple[str, str]]:
    fn = (filename or "").lower()
    try:
        if fn.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(data)):
            yield from _iter_zip(data)
            return
        if fn.endswith((".7z", ".7zip")):
            yield from _iter_7z(data)
            return
        # tar family (.tar/.tar.gz/.tgz/.tar.bz2/.tbz2/.tar.xz/.txz) — auto-detect
        yield from _iter_tar(data)
    except Exception as exc:  # noqa: BLE001 — corrupt/unsupported → nothing
        log.info("archive iteration failed for %s: %s", filename, exc)


def _iter_zip(data: bytes) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()][:_MAX_MEMBERS]
        for i in infos:
            if not _wanted(i.filename, i.file_size):
                continue
            try:
                txt = _decode(zf.read(i))
            except Exception:  # noqa: BLE001
                continue
            if txt is not None:
                yield i.filename, txt


def _iter_tar(data: bytes) -> Iterator[tuple[str, str]]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()][:_MAX_MEMBERS]
        for m in members:
            if not _wanted(m.name, int(getattr(m, "size", 0) or 0)):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            txt = _decode(f.read())
            if txt is not None:
                yield m.name, txt


def _iter_7z(data: bytes) -> Iterator[tuple[str, str]]:
    try:
        import py7zr
    except Exception:  # noqa: BLE001
        return
    with py7zr.SevenZipFile(io.BytesIO(data), mode="r") as z:
        names = [n for n in z.getnames()][:_MAX_MEMBERS]
        wanted = [n for n in names if language_for(n) is not None]
        if not wanted:
            return
        for name, bio in z.read(wanted).items():
            raw = bio.read()
            if len(raw) > _MAX_FILE_BYTES:
                continue
            txt = _decode(raw)
            if txt is not None:
                yield name, txt
