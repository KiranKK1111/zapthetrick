"""Network-condition adaptivity (perceived-speed R22, task 14.3 backend).

Pins Property 13: metered/offline suppress speculation, slow/metered prefer
compact payloads, normal restores standard behavior.
"""
from __future__ import annotations

from app.perceived import network as net


def test_suppresses_speculation_when_metered_or_offline():
    assert net.should_suppress_speculation(net.METERED) is True
    assert net.should_suppress_speculation(net.OFFLINE) is True
    assert net.should_suppress_speculation(net.SLOW) is False
    assert net.should_suppress_speculation(net.NORMAL) is False


def test_normal_restores_standard_behavior():
    assert net.should_suppress_speculation(net.NORMAL) is False
    assert net.prefers_compact_payloads(net.NORMAL) is False


def test_prefers_compact_on_constrained_links():
    assert net.prefers_compact_payloads(net.SLOW) is True
    assert net.prefers_compact_payloads(net.METERED) is True
    assert net.prefers_compact_payloads(net.OFFLINE) is True


def test_normalize_unknown_is_normal():
    assert net.normalize(None) == net.NORMAL
    assert net.normalize("garbage") == net.NORMAL
    assert net.normalize("METERED") == net.METERED
