"""Split a multi-file answer into a list of [Artifact]s.

Architecture.md §"Multi-artifact response model" says a deployment
answer should render as a tabbed card: Dockerfile, docker-compose
.yml, k8s manifest, etc. — each one downloadable. Splitting is
done by walking fenced code blocks and looking for a filename
comment at the top, or by inferring from the fence info string.

Heuristic patterns:
    ```dockerfile
    ```Dockerfile name=Dockerfile
    ```yaml file="k8s/deployment.yaml"

Anything we can't infer a filename for falls into a numbered
`snippet-N.<ext>` artifact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Artifact:
    filename: str
    language: str
    content: str


_FENCE_RE = re.compile(
    r"```([A-Za-z0-9+\-_]*)([^\n]*)\n(.*?)```",
    re.DOTALL,
)
_FILENAME_INFO_RE = re.compile(
    r"""(?:name|file|filename)\s*=\s*['"]?([^\s'"]+)""",
    re.IGNORECASE,
)
_FILENAME_COMMENT_RE = re.compile(
    r"^\s*(?://|#|--|<!--|/\*)\s*([\w./-]+\.[A-Za-z0-9]{1,8})",
)
# A path-ish token: optional dirs + a name with an extension, e.g.
# "src/components/Dashboard.tsx", "package.json", "app/main.py".
_PATHISH_RE = re.compile(r"([A-Za-z0-9_][\w.\-/]*\.[A-Za-z0-9]{1,8})")

# Extensions we treat as real source files when inferring a filename from a
# heading/sentence preceding a code block (avoids matching version numbers like
# "3.1" or hosts like "localhost:3000").
_KNOWN_EXTS = frozenset({
    "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "vue", "svelte",
    "json", "json5", "yml", "yaml", "toml", "ini", "cfg", "conf", "env",
    "html", "htm", "css", "scss", "sass", "less",
    "md", "markdown", "txt", "rst", "csv",
    "go", "rs", "java", "kt", "kts", "swift", "rb", "php", "cs",
    "c", "cc", "cpp", "cxx", "h", "hpp", "m", "mm",
    "sh", "bash", "zsh", "ps1", "bat",
    "sql", "graphql", "gql", "proto", "xml", "svg", "lock",
    "gradle", "properties", "dockerfile", "tf", "tfvars",
})


_EXT_BY_LANG: dict[str, str] = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "go": "go", "golang": "go",
    "rust": "rs",
    "java": "java",
    "kotlin": "kt",
    "swift": "swift",
    "ruby": "rb",
    "php": "php",
    "c": "c",
    "cpp": "cpp", "c++": "cpp",
    "csharp": "cs", "c#": "cs",
    "yaml": "yml", "yml": "yml",
    "json": "json",
    "dockerfile": "Dockerfile",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "tf": "tf", "terraform": "tf",
}


def split_artifacts(text: str) -> list[Artifact]:
    """Walk every fenced block, return one artifact per block.

    Filename is inferred from (in priority): the fence info-string
    (`name=…`), a first-line filename comment, or the heading / sentence
    that immediately PRECEDES the block (multi-file answers usually label
    each file with a heading like `### src/App.tsx` or `3.1 src/api.ts`).
    Anything unlabelled falls back to `snippet-N.<ext>`.
    """
    text = text or ""
    out: list[Artifact] = []
    last_end = 0
    for i, m in enumerate(_FENCE_RE.finditer(text), start=1):
        preamble = text[last_end:m.start()]
        last_end = m.end()
        lang = (m.group(1) or "").lower().strip() or "txt"
        info = m.group(2) or ""
        content = (m.group(3) or "").rstrip("\n")
        filename = _infer_filename(
            content, info, lang, preamble=preamble, idx=i
        )
        out.append(Artifact(filename=filename, language=lang, content=content))
    return out


def _infer_filename(
    content: str, info: str, lang: str, *, preamble: str = "", idx: int
) -> str:
    # 1. info-string `name=...` / `file=...`
    info_m = _FILENAME_INFO_RE.search(info or "")
    if info_m:
        return info_m.group(1)
    # 2. first-line filename comment
    first = (content.splitlines() or [""])[0]
    comment_m = _FILENAME_COMMENT_RE.match(first)
    if comment_m:
        return comment_m.group(1)
    # 3. filename in the heading / sentence right before the block
    pre = _filename_from_preamble(preamble)
    if pre:
        return pre
    # 4. fallback — snippet-N.<ext>
    ext = _EXT_BY_LANG.get(lang, "txt")
    if ext == "Dockerfile":
        return "Dockerfile" if idx == 1 else f"Dockerfile.{idx}"
    return f"snippet-{idx}.{ext}"


def _looks_like_file(cand: str) -> bool:
    """True when a path-ish token is plausibly a real filename (not a version
    number like 3.1, nor a host like example.com)."""
    base = cand.rsplit("/", 1)[-1]
    if "." not in base:
        return "/" in cand
    ext = base.rsplit(".", 1)[-1].lower()
    if ext.isdigit():
        return False
    return ext in _KNOWN_EXTS or "/" in cand


def _filename_from_preamble(pre: str) -> str:
    """Find a filename in the last few non-empty lines before a code block —
    multi-file answers label each file with a heading/path right above it."""
    lines = [ln.strip() for ln in (pre or "").splitlines() if ln.strip()]
    for ln in reversed(lines[-3:]):
        # Strip markdown chrome: heading #'s, bold/italic, backticks, list
        # bullets, and a leading "3.1"/"1)"-style section number.
        s = ln.lstrip("#").strip()
        s = s.strip("*_`").strip()
        s = re.sub(r"^\d+(?:\.\d+)*[).]?\s+", "", s)
        s = s.strip("*_`").strip()
        cands = _PATHISH_RE.findall(s)
        if not cands:
            continue
        with_slash = [c for c in cands if "/" in c]
        cand = (with_slash or cands)[-1].strip("/")
        if _looks_like_file(cand):
            return cand
    return ""


__all__ = ["Artifact", "split_artifacts"]
