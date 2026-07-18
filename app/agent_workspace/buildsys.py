"""Build-system detection (#22).

Given a workspace directory, figure out the project's toolchain and the
canonical install / build / test / lint / run commands — so the verification
loop (Phase 1) and the agent can run the *right* commands without guessing.

Deterministic, dependency-free, fast. Returns the detected systems in priority
order (a monorepo can have several). Each command is a best-effort default;
`None` means "no obvious command for this project".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class BuildSystem:
    name: str               # node | python | maven | gradle | cargo | go | dotnet | make | ruby | php
    language: str
    markers: list[str] = field(default_factory=list)  # files that triggered detection
    install: str | None = None
    build: str | None = None
    test: str | None = None
    lint: str | None = None
    run: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "language": self.language, "markers": self.markers,
            "install": self.install, "build": self.build, "test": self.test,
            "lint": self.lint, "run": self.run,
        }


def _exists(root: str, *names: str) -> list[str]:
    return [n for n in names if os.path.isfile(os.path.join(root, n))]


def _glob_exists(root: str, suffix: str) -> bool:
    try:
        return any(f.endswith(suffix) for f in os.listdir(root))
    except OSError:
        return False


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _node(root: str) -> BuildSystem | None:
    if not _exists(root, "package.json"):
        return None
    pkg = _read_json(os.path.join(root, "package.json"))
    scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
    # Prefer a lockfile-appropriate install command.
    if _exists(root, "pnpm-lock.yaml"):
        install, run_pfx = "pnpm install", "pnpm"
    elif _exists(root, "yarn.lock"):
        install, run_pfx = "yarn install", "yarn"
    else:
        install, run_pfx = "npm install", "npm run"
    has_ts = bool(_exists(root, "tsconfig.json"))
    build = (f"{run_pfx} build" if "build" in scripts
             else ("npx tsc --noEmit" if has_ts else None))
    test = (f"{'yarn' if run_pfx == 'yarn' else 'npm'} test"
            if "test" in scripts else None)
    lint = f"{run_pfx} lint" if "lint" in scripts else None
    start = f"{'yarn' if run_pfx == 'yarn' else 'npm'} start" if "start" in scripts else None
    return BuildSystem(
        name="node", language="javascript/typescript" if has_ts else "javascript",
        markers=_exists(root, "package.json", "tsconfig.json",
                        "pnpm-lock.yaml", "yarn.lock", "package-lock.json"),
        install=install, build=build, test=test, lint=lint, run=start,
    )


def _python(root: str) -> BuildSystem | None:
    markers = _exists(root, "pyproject.toml", "requirements.txt", "setup.py",
                      "setup.cfg", "Pipfile", "tox.ini")
    if not markers:
        return None
    if _exists(root, "requirements.txt"):
        install = "pip install -r requirements.txt"
    elif _exists(root, "pyproject.toml") or _exists(root, "setup.py"):
        install = "pip install -e ."
    else:
        install = "pip install ."
    # Lint: ruff if configured, else flake8 if configured.
    lint = None
    pyproject = ""
    if _exists(root, "pyproject.toml"):
        try:
            with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
                pyproject = f.read()
        except OSError:
            pyproject = ""
    if _exists(root, "ruff.toml") or "[tool.ruff]" in pyproject:
        lint = "ruff check ."
    elif _exists(root, ".flake8") or "[flake8]" in _safe_read(root, "setup.cfg"):
        lint = "flake8"
    return BuildSystem(
        name="python", language="python", markers=markers,
        install=install, build=None, test="pytest -q", lint=lint, run=None,
    )


def _safe_read(root: str, name: str) -> str:
    try:
        with open(os.path.join(root, name), encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _maven(root: str) -> BuildSystem | None:
    if not _exists(root, "pom.xml"):
        return None
    return BuildSystem(
        name="maven", language="java", markers=["pom.xml"],
        install="mvn -q -B dependency:resolve",
        build="mvn -q -B -DskipTests package", test="mvn -q -B test", run=None,
    )


def _gradle(root: str) -> BuildSystem | None:
    markers = _exists(root, "build.gradle", "build.gradle.kts", "settings.gradle",
                      "settings.gradle.kts")
    if not markers:
        return None
    g = "./gradlew" if _exists(root, "gradlew") else "gradle"
    return BuildSystem(
        name="gradle", language="java/kotlin", markers=markers,
        build=f"{g} build -x test", test=f"{g} test", run=None,
    )


def _cargo(root: str) -> BuildSystem | None:
    if not _exists(root, "Cargo.toml"):
        return None
    return BuildSystem(
        name="cargo", language="rust", markers=["Cargo.toml"],
        build="cargo build", test="cargo test", lint="cargo clippy", run="cargo run",
    )


def _go(root: str) -> BuildSystem | None:
    if not _exists(root, "go.mod"):
        return None
    return BuildSystem(
        name="go", language="go", markers=["go.mod"],
        install="go mod download", build="go build ./...",
        test="go test ./...", lint="go vet ./...", run=None,
    )


def _dotnet(root: str) -> BuildSystem | None:
    if not (_glob_exists(root, ".csproj") or _glob_exists(root, ".sln")
            or _glob_exists(root, ".fsproj")):
        return None
    return BuildSystem(
        name="dotnet", language="csharp", markers=["*.csproj/*.sln"],
        install="dotnet restore", build="dotnet build", test="dotnet test", run=None,
    )


def _ruby(root: str) -> BuildSystem | None:
    if not _exists(root, "Gemfile"):
        return None
    test = "bundle exec rspec" if os.path.isdir(os.path.join(root, "spec")) \
        else "bundle exec rake test"
    return BuildSystem(
        name="ruby", language="ruby", markers=["Gemfile"],
        install="bundle install", test=test, run=None,
    )


def _php(root: str) -> BuildSystem | None:
    if not _exists(root, "composer.json"):
        return None
    return BuildSystem(
        name="php", language="php", markers=["composer.json"],
        install="composer install", test="composer test", run=None,
    )


def _make(root: str) -> BuildSystem | None:
    markers = _exists(root, "Makefile", "makefile", "CMakeLists.txt")
    if not markers:
        return None
    if "CMakeLists.txt" in markers:
        return BuildSystem(
            name="cmake", language="c/c++", markers=markers,
            build="cmake -S . -B build && cmake --build build", run=None,
        )
    body = _safe_read(root, markers[0]).lower()
    return BuildSystem(
        name="make", language="make", markers=markers,
        build="make", test="make test" if "test:" in body else None, run=None,
    )


# Priority order: a real language toolchain wins over a bare Makefile.
_DETECTORS = (_node, _python, _maven, _gradle, _cargo, _go, _dotnet, _ruby,
              _php, _make)


def detect_build_systems(root: str) -> list[BuildSystem]:
    """Every build system detected at the workspace root, in priority order."""
    out: list[BuildSystem] = []
    if not os.path.isdir(root):
        return out
    for detector in _DETECTORS:
        try:
            bs = detector(root)
        except Exception:  # noqa: BLE001 — detection must never raise
            bs = None
        if bs is not None:
            out.append(bs)
    return out


def detect_build_system(root: str) -> BuildSystem | None:
    """The primary (highest-priority) build system, or None."""
    systems = detect_build_systems(root)
    return systems[0] if systems else None


__all__ = ["BuildSystem", "detect_build_system", "detect_build_systems"]
