r"""Verify the Docker sandbox: run a hello-world for every one of the 25
languages INSIDE the zapthetrick_sandbox container.

Prereqs:
    docker compose build sandbox      # once (~10-20 min; big toolchain image)
    docker compose up -d              # brings the sandbox up with the stack

Run (from zapthetrick_be, with the venv):
    .\.venv\Scripts\python.exe scripts\verify_sandbox_container.py

It uses the app's own run_code(), which routes to the docker backend because
config.yaml has `sandbox.backend: docker`. Any FAIL prints the compiler/runtime
error so the Dockerfile can be adjusted for that toolchain.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config_loader import cfg          # noqa: E402
from app.sandbox import docker_exec, lang_registry  # noqa: E402
from app.sandbox.executor import run_code        # noqa: E402

PROGRAMS = {
    "cpp":        '#include <iostream>\nint main(){std::cout<<"OK";return 0;}',
    "c":          '#include <stdio.h>\nint main(){printf("OK");return 0;}',
    "java":       'public class Main{public static void main(String[] a){System.out.print("OK");}}',
    "python":     'print("OK")',
    "javascript": 'console.log("OK")',
    "typescript": 'const x: string = "OK"; console.log(x)',
    "csharp":     'using System; class P{static void Main(){Console.Write("OK");}}',
    "go":         'package main\nimport "fmt"\nfunc main(){fmt.Print("OK")}',
    "kotlin":     'fun main(){print("OK")}',
    "swift":      'print("OK")',
    "rust":       'fn main(){print!("OK");}',
    "ruby":       'print "OK"',
    "php":        '<?php echo "OK"; ?>',
    "dart":       'void main(){print("OK");}',
    "scala":      'object Main extends App{print("OK")}',
    "elixir":     'IO.write("OK")',
    "erlang":     '-module(main).\n-export([main/1]).\nmain(_) -> io:format("OK").',
    "racket":     '#lang racket\n(display "OK")',
    "sql":        "SELECT 'OK';",
    "bash":       'echo OK',
    "r":          'cat("OK")',
    "julia":      'print("OK")',
    "perl":       'print "OK";',
    "groovy":     'print "OK"',
    # --- Tier 2 -----------------------------------------------------------
    "ada":        'with Ada.Text_IO; use Ada.Text_IO;\nprocedure Main is\nbegin\n  Put("OK");\nend Main;',
    "awk":        'BEGIN{printf "OK"}',
    "clojure":    '(print "OK")',
    "cobol":      ('       IDENTIFICATION DIVISION.\n'
                   '       PROGRAM-ID. MAIN.\n'
                   '       PROCEDURE DIVISION.\n'
                   '           DISPLAY "OK".\n'
                   '           STOP RUN.'),
    "crystal":    'print "OK"',
    "d":          'import std.stdio; void main(){write("OK");}',
    "fish":       'echo -n OK',
    "fortran":    "program main\n  write(*,'(A)',advance='no') 'OK'\nend program main",
    "fsharp":     'printf "OK"',
    "haskell":    'main = putStr "OK"',
    "hy":         '(print "OK")',
    "lisp":       '(princ "OK")',
    "lua":        'io.write("OK")',
    "nim":        'stdout.write("OK")',
    "objc":       '#include <stdio.h>\nint main(){printf("OK");return 0;}',
    "ocaml":      'print_string "OK"',
    "octave":     'printf("OK")',
    "odin":       'package main\nimport "core:fmt"\nmain :: proc() { fmt.print("OK") }',
    "pascal":     "begin write('OK') end.",
    "powershell": 'Write-Output "OK"',
    "prolog":     "main :- write('OK').",
    "raku":       'print "OK"',
    "scheme":     '(display "OK")',
    "smalltalk":  "Transcript show: 'OK'.",
    "sml":        'print "OK";',
    "tcl":        'puts -nonewline OK',
    "vlang":      'fn main(){print("OK")}',
    "zig":        ('const std = @import("std");\n'
                   'pub fn main() !void {\n'
                   '    try std.io.getStdOut().writer().print("OK", .{});\n}'),
    "zsh":        'print -n OK',
}


def main() -> int:
    print(f"backend         = {cfg.sandbox.backend}")
    print(f"container       = {cfg.sandbox.container}")
    print(f"docker present  = {docker_exec._docker_bin() is not None}")
    print(f"container up    = {docker_exec.available(refresh=True)}")
    if not docker_exec.available():
        print("\n!! Sandbox container is not running. Build + start it first:\n"
              "     docker compose build sandbox\n"
              "     docker compose up -d\n")
        return 2
    print()

    ok, bad = [], []
    for lang, code in PROGRAMS.items():
        res = run_code(code, lang)
        good = res.ok and "OK" in (res.stdout or "")
        tag = "PASS" if good else "FAIL"
        detail = "" if good else (
            f"  status={res.status} rc={res.exit_code} "
            f"reason={(res.reason or '')[:80]} "
            f"err={(res.stderr or '')[:160]}")
        print(f"[{tag}] {lang:11}{detail}")
        (ok if good else bad).append(lang)

    print(f"\n=== {len(ok)}/{len(PROGRAMS)} PASS ===")
    if bad:
        print("FAIL:", ", ".join(bad))

    # --- multiple toolchain versions ------------------------------------
    print("\n--- version pins ---")
    # (lang, version, version-appropriate hello). Python 2 uses the print
    # STATEMENT, which only parses on 2.x — a strong version discriminator.
    vtests = [
        ("python", "2.7", 'print "OK"'),
        ("python", "3.12", 'print("OK")'),
        ("python", "3.14", 'print("OK")'),
        ("javascript", "18", 'console.log("OK")'),
        ("javascript", "22", 'console.log("OK")'),
        ("java", "8", 'public class Main{public static void main(String[] a){System.out.print("OK");}}'),
        ("java", "21", 'public class Main{public static void main(String[] a){System.out.print("OK");}}'),
    ]
    vbad = []
    for lang, ver, code in vtests:
        res = run_code(code, lang, version=ver)
        good = res.ok and "OK" in (res.stdout or "")
        print(f"[{'PASS' if good else 'FAIL'}] {lang}@{ver}"
              + ("" if good else f"  {res.status} {(res.stderr or res.reason or '')[:80]}"))
        if not good:
            vbad.append(f"{lang}@{ver}")

    # --- small multi-file project ---------------------------------------
    print("\n--- multi-file project ---")
    proj = run_code('from helper import msg\nprint(msg())', "python",
                    files={"helper.py": 'def msg():\n    return "OK"\n'})
    proj_ok = proj.ok and "OK" in (proj.stdout or "")
    print(f"[{'PASS' if proj_ok else 'FAIL'}] python 2-file import"
          + ("" if proj_ok else f"  {proj.status} {(proj.stderr or proj.reason or '')[:80]}"))

    return 0 if (not bad and not vbad and proj_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
