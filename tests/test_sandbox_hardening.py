"""Sandbox hardening (P4 #19): static escape/egress scan + backend-aware block."""
from __future__ import annotations

from app.sandbox import hardening


def test_flags_socket_egress():
    f = hardening.scan("import socket\ns = socket.socket()")
    assert any(x.category == "egress" and x.severity == 3 for x in f)


def test_flags_http_and_shell_nettools():
    assert any(x.category == "egress"
               for x in hardening.scan("import requests; requests.get('http://x')"))
    assert any(x.category == "egress"
               for x in hardening.scan("import os; os.system('curl http://evil')"))


def test_flags_secret_reads():
    assert any(x.category == "secret"
               for x in hardening.scan("open('/etc/shadow').read()"))


def test_comments_do_not_trip_scanner():
    assert hardening.scan("# do not use socket.socket() here\nprint(1)") == []


def test_clean_script_has_no_findings():
    assert hardening.scan("print(sum(range(10)))") == []


def test_blocked_only_on_non_isolated_backend():
    code = "import socket; socket.socket()"
    # namespace backend contains network → advisory, not blocked
    assert hardening.assess(code, net_isolated=True).blocked is False
    # rlimit/subprocess can't isolate net → blocked
    r = hardening.assess(code, net_isolated=False)
    assert r.blocked is True and r.max_severity == 3
