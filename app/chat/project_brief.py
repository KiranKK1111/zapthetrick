"""Build a short, accurate brief for a downloadable project ZIP.

When the user asks to "zip the project", the fast-path doesn't call the LLM
again — instead it extracts, from the project the model already produced in the
conversation, a proper project NAME, a one-line OVERVIEW, and HOW-TO-RUN steps,
and composes a concise confirmation message shown above the Download card.

Everything here is deterministic string parsing — no LLM, so it's instant.
"""
from __future__ import annotations

import re

_TREE_ROOT_RE = re.compile(r"^\s*([A-Za-z0-9][\w.\-]*)/\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
# Commands that look like real install/run steps (not curl examples etc.).
_RUN_CMD_RE = re.compile(
    r"\b("
    r"npm (?:install|ci|run [\w:-]+|start)|yarn(?: install| dev| start| build)?|"
    r"pnpm [\w:-]+|bun (?:install|run [\w:-]+)|"
    r"pip install[^\n]*|pip3 install[^\n]*|poetry install|"
    r"uvicorn [\w.:]+[^\n]*|gunicorn [^\n]*|flask run[^\n]*|"
    r"python3? -m [^\n]*|python3? [\w./]+\.py|"
    r"go run [^\n]*|cargo run[^\n]*|"
    r"docker compose up[^\n]*|docker-compose up[^\n]*|docker run[^\n]*|"
    r"\./mvnw[^\n]*|mvn [^\n]*|\./gradlew[^\n]*|gradle [^\n]*|"
    r"dotnet run[^\n]*|rails server|bundle exec[^\n]*"
    r")",
    re.IGNORECASE,
)


def project_name(content: str) -> str:
    """A clean project name from the directory-tree root folder, else a title."""
    m = _TREE_ROOT_RE.search(content or "")
    if m:
        return m.group(1)
    for h in _HEADING_RE.findall(content or ""):
        t = re.sub(r"[^\w \-]", "", h).strip()
        # Skip generic section headings.
        if t and t.lower() not in {
            "project overview", "overview", "folder structure", "core files",
            "core code", "directory layout", "summary", "how to run",
        }:
            return re.sub(r"\s+", "_", t)[:40]
    return "project"


def overview(content: str) -> str:
    """First substantial prose sentence(s) describing the project."""
    # Prefer the paragraph right after an "Overview"-style heading.
    m = re.search(
        r"(?:project overview|overview)\s*\n+([^\n#`].+?)(?:\n\s*\n|\n#|\Z)",
        content or "", re.IGNORECASE | re.DOTALL,
    )
    if m:
        return " ".join(m.group(1).split())[:400]
    for para in re.split(r"\n\s*\n", content or ""):
        s = para.strip()
        if (not s or s.startswith("#") or s.startswith("```")
                or any(c in s for c in "├└│")):
            continue
        if len(s) >= 40:
            return " ".join(s.split())[:400]
    return ""


def _run_from_blocks(content: str) -> list[str]:
    """Pull real install/run commands out of the answer's shell blocks."""
    cmds: list[str] = []
    for m in re.finditer(r"```(?:bash|sh|shell|console|zsh)?\s*\n(.*?)```",
                         content or "", re.DOTALL):
        for line in m.group(1).splitlines():
            line = line.strip().lstrip("$ ").strip()
            if _RUN_CMD_RE.search(line) and line not in cmds:
                cmds.append(line)
    return cmds[:6]


def run_steps(content: str) -> list[str]:
    """How-to-run steps as short strings (commands wrapped in `backticks`)."""
    c = (content or "").lower()
    cmds = _run_from_blocks(content)
    steps: list[str] = ["Unzip the archive."]
    if cmds:
        steps += [f"`{cmd}`" for cmd in cmds]
        return steps
    # No explicit commands found — infer from the stack.
    if "package.json" in c:
        steps += ["`npm install`", "`npm run dev`",
                  "Open the printed URL (usually `http://localhost:3000`)."]
    elif "requirements.txt" in c or "uvicorn" in c or "fastapi" in c:
        steps += ["`python -m venv .venv` and activate it",
                  "`pip install -r requirements.txt`"]
        if "uvicorn" in c or "fastapi" in c:
            steps += ["`uvicorn app.main:app --reload`",
                      "Open `http://localhost:8000/docs`"]
        elif "flask" in c:
            steps += ["`flask run`"]
        else:
            steps += ["Run the entry script, e.g. `python main.py`"]
    elif "go.mod" in c or "cargo.toml" in c:
        steps += ["Build/run with your toolchain (`go run .` / `cargo run`)."]
    else:
        steps += ["Install the dependencies for the stack used.",
                  "Run the project's entry point."]
    return steps


def build_brief(content: str) -> tuple[str, str]:
    """Return (project_name, markdown_message) for the ZIP confirmation."""
    name = project_name(content)
    ov = overview(content)
    steps = run_steps(content)
    lines = [f"**{name}** is packaged and ready to download."]
    if ov:
        lines.append("")
        lines.append(ov)
    lines.append("")
    lines.append("**How to run it after download:**")
    for i, s in enumerate(steps, start=1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("Use the **Download** button below to save the ZIP archive.")
    return name, "\n".join(lines)


__all__ = ["build_brief", "project_name", "overview", "run_steps"]
