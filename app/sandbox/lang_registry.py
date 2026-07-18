"""Data-driven language registry for the sandbox (SandboxLangPack.md).

Rather than hardcode execution per language across the codebase, every language
is ONE data entry here: its source filename, the tool that must be on PATH, and
its build/run commands. Adding a language = adding a row. The sandbox executes
whatever runtimes are actually INSTALLED on the host; a language whose toolchain
is absent is reported unavailable (the model can still WRITE it, we just can't
run/verify it) — no crash.

Covers Tier 1 (~15) + Tier 2 (~20) of the doc. Placeholders resolved per host:
  {main}  → the source filename       {class} → Java/Kotlin main class from code
  {exe}   → OS-aware run path (./prog | prog.exe)   {out} → compiled binary name
"""
from __future__ import annotations

import os
import re
import shutil
import sys

_PY = sys.executable or "python3"

_SEARCH_PATH: str | None = None
_EXTRA_DIRS: list[str] | None = None


def _extra_dirs() -> list[str]:
    """`sandbox.tool_dirs` (bin dirs a manually-installed toolchain lives in,
    searched first). Cached. Kept in sync with the executor's copy so the
    availability check and the actual run resolve the SAME binary."""
    global _EXTRA_DIRS
    if _EXTRA_DIRS is not None:
        return _EXTRA_DIRS
    dirs: list[str] = []
    try:
        from app.core.config_loader import cfg
        for d in (getattr(cfg.sandbox, "tool_dirs", None) or []):
            if d and os.path.isdir(str(d)):
                dirs.append(str(d))
    except Exception:  # noqa: BLE001
        pass
    _EXTRA_DIRS = dirs
    return dirs


def search_path() -> str:
    """The PATH used to resolve runtimes: `sandbox.tool_dirs` + the CURRENTLY
    persisted system PATH + this process's PATH.

    A long-running server's `os.environ['PATH']` is a snapshot from launch time,
    so a toolchain installed (and PATH-registered) afterwards is invisible to it
    — that's why Ruby/Erlang showed "runtime not installed" though the sandbox
    could run them. On Windows we read the live Machine+User PATH from the
    registry so newly-installed tools resolve WITHOUT a server restart. Cached
    per process (a restart re-reads)."""
    global _SEARCH_PATH
    if _SEARCH_PATH is not None:
        return _SEARCH_PATH
    parts = list(_extra_dirs())
    if os.name == "nt":
        try:
            import winreg
            for root, sub in (
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, "Environment"),
            ):
                try:
                    with winreg.OpenKey(root, sub) as k:
                        val, _ = winreg.QueryValueEx(k, "Path")
                        if val:
                            parts.append(os.path.expandvars(val))
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("PATH"):
        parts.append(os.environ["PATH"])
    _SEARCH_PATH = os.pathsep.join(p for p in parts if p)
    return _SEARCH_PATH


def _exe() -> str:
    return "prog.exe" if os.name == "nt" else "./prog"


def _out() -> str:
    return "prog.exe" if os.name == "nt" else "prog"


def _jvm_class(code: str) -> str:
    """Public class name so the file can be <Class>.java / run as `java <Class>`."""
    m = (re.search(r"public\s+(?:final\s+|abstract\s+)?class\s+([A-Za-z_]\w*)", code or "")
         or re.search(r"\bclass\s+([A-Za-z_]\w*)", code or ""))
    return m.group(1) if m else "Main"


# id -> spec. `check` is the executable that must exist on PATH.
# Interpreted: only `run`. Compiled: `build` then `run`.
_REGISTRY: dict[str, dict] = {
    # ── Tier 1 ────────────────────────────────────────────────────────────
    "python": {"aliases": ["py", "python3", "python2", "py3", "py2"],
               "file": "main.py", "check": "PYEXE",
               "run": [_PY, "-I", "-B", "main.py"]},
    "java": {"aliases": [], "file": "{class}.java", "check": "javac",
             "build": ["javac", "{main}"], "run": ["java", "{class}"]},
    "javascript": {"aliases": ["js", "node", "nodejs"], "file": "main.js",
                   "check": "node", "run": ["node", "main.js"]},
    "typescript": {"aliases": ["ts"], "file": "main.ts", "check": "tsx",
                   "run": ["tsx", "main.ts"],
                   "alts": [{"check": "ts-node", "run": ["ts-node", "main.ts"]},
                            {"check": "deno", "run": ["deno", "run", "-A",
                                                      "main.ts"]}]},
    "c": {"aliases": [], "file": "main.c", "check": "gcc",
          "build": ["gcc", "-O2", "main.c", "-o", "{out}"], "run": ["{exe}"]},
    "cpp": {"aliases": ["c++", "cxx", "cc"], "file": "main.cpp", "check": "g++",
            "build": ["g++", "-O2", "-std=c++17", "main.cpp", "-o", "{out}"],
            "run": ["{exe}"]},
    # Primary: .NET 10+ file-based apps (`dotnet run main.cs`, no project);
    # fallbacks: Roslyn `csc` compile, or Mono `mcs`.
    "csharp": {"aliases": ["cs", "c#", "dotnet"], "file": "main.cs",
               "check": "dotnet", "run": ["dotnet", "run", "main.cs"],
               "alts": [{"check": "csc", "build": ["csc", "-nologo", "main.cs"],
                         "run": ["{exe}"]},
                        {"check": "mcs", "build": ["mcs", "main.cs"],
                         "run": ["mono", "main.exe"]}]},
    "go": {"aliases": ["golang"], "file": "main.go", "check": "go",
           "run": ["go", "run", "main.go"]},
    "rust": {"aliases": ["rs"], "file": "main.rs", "check": "rustc",
             "build": ["rustc", "-O", "main.rs", "-o", "{out}"], "run": ["{exe}"]},
    "kotlin": {"aliases": ["kt"], "file": "main.kt", "check": "kotlinc",
               "build": ["kotlinc", "main.kt", "-include-runtime", "-d", "main.jar"],
               "run": ["java", "-jar", "main.jar"]},
    # `swift main.swift` (JIT interpret) can't resolve stdlib symbols on
    # Windows ("JIT session error: Symbols not found"); compiling with swiftc
    # to a native exe works once SDKROOT is set (executor supplies it).
    "swift": {"aliases": [], "file": "main.swift", "check": "swiftc",
              "build": ["swiftc", "main.swift", "-o", "{out}"],
              "run": ["{exe}"]},
    "php": {"aliases": [], "file": "main.php", "check": "php",
            "run": ["php", "main.php"]},
    "ruby": {"aliases": ["rb"], "file": "main.rb", "check": "ruby",
             "run": ["ruby", "main.rb"]},
    "r": {"aliases": ["rlang"], "file": "main.R", "check": "Rscript",
          "run": ["Rscript", "main.R"]},
    "bash": {"aliases": ["sh", "shell"], "file": "main.sh", "check": "bash",
             "run": ["bash", "main.sh"]},
    # ── Tier 2 ────────────────────────────────────────────────────────────
    "dart": {"aliases": [], "file": "main.dart", "check": "dart",
             "run": ["dart", "run", "main.dart"]},
    "scala": {"aliases": [], "file": "main.scala", "check": "scala",
              "run": ["scala", "main.scala"]},
    "lua": {"aliases": [], "file": "main.lua", "check": "lua",
            "run": ["lua", "main.lua"]},
    "perl": {"aliases": ["pl"], "file": "main.pl", "check": "perl",
             "run": ["perl", "main.pl"]},
    "julia": {"aliases": ["jl"], "file": "main.jl", "check": "julia",
              "run": ["julia", "main.jl"]},
    "elixir": {"aliases": ["exs"], "file": "main.exs", "check": "elixir",
               "run": ["elixir", "main.exs"]},
    # escript runs a `.erl` module directly IF it declares -module(main) +
    # -export([main/1]); the verify harness is prompted to emit exactly that.
    "erlang": {"aliases": ["erl"], "file": "main.erl", "check": "escript",
               "run": ["escript", "main.erl"]},
    "haskell": {"aliases": ["hs"], "file": "main.hs", "check": "runghc",
                "run": ["runghc", "main.hs"]},
    "ocaml": {"aliases": ["ml"], "file": "main.ml", "check": "ocaml",
              "run": ["ocaml", "main.ml"]},
    "fsharp": {"aliases": ["fs", "f#", "fsx"], "file": "main.fsx",
               "check": "dotnet", "run": ["dotnet", "fsi", "main.fsx"]},
    "nim": {"aliases": [], "file": "main.nim", "check": "nim",
            "build": ["nim", "c", "-d:release", "--out:{out}", "main.nim"],
            "run": ["{exe}"]},
    "zig": {"aliases": [], "file": "main.zig", "check": "zig",
            "run": ["zig", "run", "main.zig"]},
    "fortran": {"aliases": ["f90", "f95"], "file": "main.f90", "check": "gfortran",
                "build": ["gfortran", "main.f90", "-o", "{out}"], "run": ["{exe}"]},
    "groovy": {"aliases": [], "file": "main.groovy", "check": "groovy",
               "run": ["groovy", "main.groovy"]},
    "octave": {"aliases": ["matlab", "m"], "file": "main.m", "check": "octave",
               "run": ["octave", "--no-gui", "-q", "main.m"]},
    "sql": {"aliases": ["sqlite", "sqlite3"], "file": "main.sql", "check": "sqlite3",
            "run": ["sqlite3", ":memory:", ".read main.sql"]},
    "powershell": {"aliases": ["ps1", "pwsh"], "file": "main.ps1",
                   "check": "powershell", "run": ["powershell", "-NoProfile",
                   "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                   "main.ps1"]},
    "tcl": {"aliases": [], "file": "main.tcl", "check": "tclsh",
            "run": ["tclsh", "main.tcl"]},
    # ── Tier 3 / extra LeetCode & HackerRank coverage ─────────────────────
    "racket": {"aliases": ["rkt"], "file": "main.rkt", "check": "racket",
               "run": ["racket", "main.rkt"]},
    # Babashka (`bb`) is a fast, self-contained Clojure interpreter — the most
    # reliable Clojure runner on Windows (no JVM project setup).
    "clojure": {"aliases": ["clj"], "file": "main.clj", "check": "bb",
                "run": ["bb", "main.clj"],
                "alts": [{"check": "clojure", "run": ["clojure", "main.clj"]},
                         {"check": "clj", "run": ["clj", "main.clj"]}]},
    "lisp": {"aliases": ["commonlisp", "cl"], "file": "main.lisp",
             "check": "sbcl", "run": ["sbcl", "--script", "main.lisp"]},
    "scheme": {"aliases": ["guile"], "file": "main.scm", "check": "guile",
               "run": ["guile", "main.scm"]},
    "prolog": {"aliases": ["swipl"], "file": "main.pro", "check": "swipl",
               "run": ["swipl", "-q", "-g", "main", "-t", "halt", "main.pro"]},
    "pascal": {"aliases": ["pas", "delphi"], "file": "main.pas", "check": "fpc",
               # -o{out}: fpc names the exe after the SOURCE by default (`main`),
               # but the run step expects `{exe}` (prog) — pin the output name.
               "build": ["fpc", "-o{out}", "main.pas"], "run": ["{exe}"]},
    "cobol": {"aliases": ["cob"], "file": "main.cob", "check": "cobc",
              "build": ["cobc", "-x", "main.cob", "-o", "{out}"], "run": ["{exe}"]},
    "d": {"aliases": ["dlang"], "file": "main.d", "check": "dmd",
          "build": ["dmd", "main.d", "-of{out}"], "run": ["{exe}"],
          "alts": [{"check": "ldc2", "build": ["ldc2", "main.d", "-of={out}"],
                    "run": ["{exe}"]}]},
    "crystal": {"aliases": ["cr"], "file": "main.cr", "check": "crystal",
                "run": ["crystal", "main.cr"]},
    "vlang": {"aliases": ["v"], "file": "main.v", "check": "v",
              "run": ["v", "run", "main.v"]},
    # ── More systems / functional / scripting / .NET / niche ──────────────
    "objc": {"aliases": ["objectivec", "objective-c", "objc"], "file": "main.m",
             "check": "clang",
             "build": ["clang", "main.m", "-o", "{out}", "-lobjc"],
             "run": ["{exe}"]},
    "vbnet": {"aliases": ["vb", "visualbasic", "vb.net"], "file": "main.vb",
              "check": "vbc", "build": ["vbc", "-nologo", "main.vb"],
              "run": ["{exe}"]},
    "awk": {"aliases": ["gawk"], "file": "main.awk", "check": "awk",
            "run": ["awk", "-f", "main.awk"]},
    "fish": {"aliases": [], "file": "main.fish", "check": "fish",
             "run": ["fish", "main.fish"]},
    "zsh": {"aliases": [], "file": "main.zsh", "check": "zsh",
            "run": ["zsh", "main.zsh"]},
    "odin": {"aliases": [], "file": "main.odin", "check": "odin",
             "run": ["odin", "run", "main.odin", "-file"]},
    "ada": {"aliases": ["adb"], "file": "main.adb", "check": "gnatmake",
            "build": ["gnatmake", "-o", "{out}", "main.adb"], "run": ["{exe}"]},
    "sml": {"aliases": ["standardml", "smlnj"], "file": "main.sml",
            "check": "sml", "run": ["sml", "main.sml"]},
    "smalltalk": {"aliases": ["gst", "st"], "file": "main.st", "check": "gst",
                  "run": ["gst", "main.st"]},
    "mojo": {"aliases": [], "file": "main.mojo", "check": "mojo",
             "run": ["mojo", "run", "main.mojo"]},
    "hack": {"aliases": ["hhvm"], "file": "main.hack", "check": "hhvm",
             "run": ["hhvm", "main.hack"]},
    # Hy is a Python package (a console script that may not be on PATH) — run it
    # via the interpreter module so it works wherever Python does.
    "hy": {"aliases": [], "file": "main.hy", "check": "PYEXE",
           "run": [_PY, "-m", "hy", "main.hy"]},
    "red": {"aliases": [], "file": "main.red", "check": "red",
            "run": ["red", "main.red"]},
    "chapel": {"aliases": ["chpl"], "file": "main.chpl", "check": "chpl",
               "build": ["chpl", "main.chpl", "-o", "{out}"], "run": ["{exe}"]},
    "gleam": {"aliases": [], "file": "main.gleam", "check": "gleam",
              "run": ["gleam", "run", "-m", "main"]},
    "raku": {"aliases": ["perl6"], "file": "main.raku", "check": "raku",
             "run": ["raku", "main.raku"]},
    "make": {"aliases": ["makefile"], "file": "Makefile", "check": "make",
             "run": ["make", "-s"]},
    # Compile/validate-only (no standalone execution model): a clean compile is
    # the pass signal.
    "solidity": {"aliases": ["sol"], "file": "main.sol", "check": "solc",
                 "run": ["solc", "--bin", "main.sol"]},
    "elm": {"aliases": [], "file": "main.elm", "check": "elm",
            "run": ["elm", "make", "main.elm", "--output=out.js"]},
}

# alias -> canonical id (built once).
_ALIAS: dict[str, str] = {}
for _cid, _spec in _REGISTRY.items():
    _ALIAS[_cid] = _cid
    for _a in _spec.get("aliases", []):
        _ALIAS[_a] = _cid


def _tool_ok(tool: str) -> bool:
    return tool == _PY or shutil.which(tool, path=search_path()) is not None


def _effective(cid: str) -> dict:
    """The variant of a language spec that's actually INSTALLED — primary if its
    tool is present, else the first available alternate (e.g. TypeScript falls
    tsx → ts-node → deno; C# falls dotnet → csc → mcs). Returns the primary spec
    when nothing is available (so `plan` still yields a command and
    `is_available` reports False)."""
    spec = _REGISTRY[cid]
    prim = _PY if spec["check"] == "PYEXE" else spec["check"]
    if _tool_ok(prim):
        return spec
    for alt in spec.get("alts", []):
        if _tool_ok(alt["check"]):
            eff = {"file": spec["file"], "check": alt["check"],
                   "run": alt["run"]}
            if "build" in alt:
                eff["build"] = alt["build"]
            return eff
    return spec


def canonical(lang: str | None) -> str | None:
    """Resolve any name/alias to a registry id (e.g. 'c++'→'cpp'). None if
    unknown."""
    if not lang:
        return None
    key = re.sub(r"[^a-z0-9+#]", "", lang.lower())
    if key in _ALIAS:
        return _ALIAS[key]
    # tolerant: python3.11 → python, etc.
    if key.startswith("python") or key in ("py", "py3", "py2"):
        return "python"
    return None


def check_tool(lang: str) -> str | None:
    """The executable that must be on PATH for `lang` (canonical or alias) —
    the installed variant when one exists, else the primary."""
    cid = canonical(lang)
    if cid is None:
        return None
    tool = _effective(cid)["check"]
    return _PY if tool == "PYEXE" else tool


def is_available(lang: str) -> bool:
    """True when `lang`'s toolchain (primary OR an alternate) is installed."""
    cid = canonical(lang)
    if cid is None:
        return False
    eff = _effective(cid)
    tool = _PY if eff["check"] == "PYEXE" else eff["check"]
    if not _tool_ok(tool):
        return False
    # Compiled/JVM langs may need a second tool (kotlin/java → `java` to run).
    run0 = eff["run"][0]
    if run0 not in ("{exe}",) and run0 != tool:
        return _tool_ok(run0)
    return True


def supported_ids() -> list[str]:
    """Every language id in the registry (installed or not)."""
    return list(_REGISTRY.keys())


def available_languages() -> list[str]:
    """Registry ids whose toolchain is actually installed on this host."""
    return [cid for cid in _REGISTRY if is_available(cid)]


# Multiple toolchain MAJOR VERSIONS installed in the sandbox container, so a run
# can be pinned to e.g. Python 2.7 vs 3.14. `default` is what a version-less run
# uses (it maps to the plain `python3`/`node`/`javac` the base plan already
# emits — the container symlinks/PATH make those the default version). Extend by
# adding a version here AND installing it in sandbox/Dockerfile. Everything not
# listed is single-version (its major rarely matters for a standalone program).
LANG_VERSIONS: dict = {
    # default = the version the plain `python3`/`node`/`javac` resolves to in the
    # container (kept as the system Python 3.10 so the pip-installed hy tooling
    # keeps working); the rest are explicit pins.
    "python":     {"default": "3.10", "all": ["2.7", "3.10", "3.12", "3.14"]},
    "javascript": {"default": "20",   "all": ["18", "20", "22"]},
    "java":       {"default": "17",   "all": ["8", "17", "21"]},
}


def available_versions(lang: str) -> list[str]:
    """Major versions the container carries for `lang` (empty = single-version)."""
    cid = canonical(lang)
    return list(LANG_VERSIONS.get(cid, {}).get("all", [])) if cid else []


def default_version(lang: str) -> str | None:
    cid = canonical(lang)
    return LANG_VERSIONS.get(cid, {}).get("default") if cid else None


def version_from_label(label: str | None) -> str | None:
    """Pull a container toolchain version out of a language label:
    'Python2'→'2.7', 'Python 3.14'→'3.14', 'Node 18'→'18'. None → default (a
    label with no/unknown version, or a single-version language)."""
    if not label:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", label)
    if not m:
        return None
    v = m.group(1)
    # Canonicalize the NON-numeric part — "Node 18"/"Java 8"/"Python2" would
    # otherwise collapse to "node18"/"java8"/"python2", which aren't aliases.
    lang_part = re.sub(r"[\d.]+", " ", label).strip() or label
    cid = canonical(lang_part)
    if not cid or cid not in LANG_VERSIONS:
        return None
    avail = LANG_VERSIONS[cid]["all"]
    if v in avail:               # exact "3.14" / "18"
        return v
    for a in avail:              # bare major: "2"→2.7, "3"→3.10, "8"→8
        if a.split(".")[0] == v:
            return a
    return None


def _versioned_cmds(cid: str, version: str, fname: str, cls: str
                    ) -> list[list[str]] | None:
    """The compile/run argv for a SPECIFIC installed version, or None if the
    language/version isn't a known multi-version pair. Major versions can differ
    in flags (Python 2 has no `-I`) and in the binary/JDK path, so each is built
    explicitly rather than by string-swapping the default plan."""
    if cid == "python":
        b = {"2.7": "python2.7", "3.12": "python3.12",
             "3.14": "python3.14"}.get(version)
        if not b:
            return None
        # -B only (no -I): the container already isolates, and -I strips the
        # script's dir from sys.path — which breaks sibling imports in a small
        # multi-file project.
        return [[b, "-B", fname]]
    if cid == "javascript":
        b = {"18": "node18", "20": "node20", "22": "node22"}.get(version)
        return [[b, fname]] if b else None
    if cid == "java":
        jdk = {"8": "/usr/lib/jvm/java-8-openjdk-amd64",
               "17": "/usr/lib/jvm/java-17-openjdk-amd64",
               "21": "/usr/lib/jvm/java-21-openjdk-amd64"}.get(version)
        if not jdk:
            return None
        return [[f"{jdk}/bin/javac", fname], [f"{jdk}/bin/java", cls]]
    return None


def plan(lang: str, code: str, *, posix: bool = False, version: str | None = None
         ) -> tuple[str, list[list[str]]] | None:
    """(source filename, [argv, ...]) for `lang`. Interpreted → one run cmd;
    compiled → [build, run]. None when the language isn't in the registry.

    `posix=True` targets the LINUX sandbox CONTAINER (docker backend): use the
    POSIX exe forms (`./prog` / `prog`), resolve the Python interpreter token to
    `python3`, and always use the PRIMARY toolchain (the container installs the
    primary tools — host-based variant detection via `_effective` doesn't apply
    inside it).

    `version` pins a specific installed major version (e.g. Python "2.7") for the
    languages in `LANG_VERSIONS`; ignored/None → the container's default. Only
    meaningful with `posix=True` (the versions live in the container)."""
    cid = canonical(lang)
    if cid is None:
        return None
    spec = (_REGISTRY.get(cid) if posix else _effective(cid))
    if spec is None:
        return None
    cls = _jvm_class(code) if cid in ("java", "kotlin") else "Main"
    fname = spec["file"].replace("{class}", cls)

    # Version pin (container only): swap in the versioned toolchain commands.
    if posix and version and cid in LANG_VERSIONS \
            and version != LANG_VERSIONS[cid].get("default"):
        vcmds = _versioned_cmds(cid, version, fname, cls)
        if vcmds is not None:
            return fname, vcmds
    _ex = "./prog" if posix else _exe()
    _ou = "prog" if posix else _out()
    _py = "python3" if posix else _PY

    def _resolve(argv: list[str]) -> list[str]:
        out = []
        for tok in argv:
            # The Python run token is the host interpreter path (`_PY`); in the
            # container it must be `python3`.
            if tok == _PY:
                out.append(_py)
                continue
            tok = (tok.replace("{main}", fname).replace("{class}", cls)
                      .replace("{exe}", _ex).replace("{out}", _ou))
            out.append(tok)
        return out

    cmds: list[list[str]] = []
    if "build" in spec:
        cmds.append(_resolve(spec["build"]))
    cmds.append(_resolve(spec["run"]))
    # Container Python: drop -I (isolated mode) — it removes the script's dir
    # from sys.path, breaking sibling imports in a small multi-file project. The
    # container already provides the isolation -I was there for.
    if posix and cid == "python":
        cmds = [[t for t in c if t != "-I"] for c in cmds]
    return fname, cmds


# Canonical ids the sandbox CONTAINER (sandbox/Dockerfile) has toolchains for —
# the 25 runnable languages. In docker-only mode, availability = this set (the
# container is authoritative; the Windows host's PATH is irrelevant).
CONTAINER_LANGS: frozenset = frozenset({
    # Tier 1 — LeetCode dropdown + interview helpers (25 dropdown / 24 runtimes).
    "cpp", "c", "java", "python", "javascript", "typescript", "csharp", "go",
    "kotlin", "swift", "rust", "ruby", "php", "dart", "scala", "elixir",
    "erlang", "racket", "sql", "bash", "r", "julia", "perl", "groovy",
    # Tier 2 — extra toolchains the Linux image also carries (2026-07-17).
    "ada", "awk", "clojure", "cobol", "crystal", "d", "fish", "fortran",
    "fsharp", "haskell", "hy", "lisp", "lua", "nim", "objc", "ocaml", "octave",
    "odin", "pascal", "powershell", "prolog", "raku", "scheme", "smalltalk",
    "sml", "tcl", "vlang", "zig", "zsh",
})


def container_supports(lang: str) -> bool:
    """True when `lang` (canonical or alias) is one the sandbox container runs."""
    cid = canonical(lang)
    return bool(cid and cid in CONTAINER_LANGS)


# Compiled languages whose compiler takes EXPLICIT source files — so a small
# multi-file project links. (Rust/others resolve modules by declaration and need
# no command change; interpreted languages import at runtime.)
_MULTIFILE_EXT: dict = {
    "c": (".c",), "cpp": (".cpp", ".cc", ".cxx"), "java": (".java",),
    "go": (".go",),
}


def augment_multifile(lang: str, main_name: str,
                      commands: list[list[str]],
                      files: dict[str, str] | None) -> list[list[str]]:
    """Fold extra staged SOURCE files of the same language into the compile step
    (e.g. `gcc main.c helper.c`, `javac Main.java Helper.java`, `go run main.go
    helper.go`) so a small multi-file project builds. No-op otherwise."""
    cid = canonical(lang)
    exts = _MULTIFILE_EXT.get(cid or "")
    if not exts or not files:
        return commands
    extra = [rel for rel in files
             if rel != main_name and rel.lower().endswith(tuple(exts))]
    if not extra:
        return commands
    out = []
    for cmd in commands:
        if main_name in cmd:
            idx = cmd.index(main_name)
            cmd = cmd[:idx + 1] + extra + cmd[idx + 1:]
        out.append(cmd)
    return out
