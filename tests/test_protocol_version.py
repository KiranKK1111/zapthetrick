"""FE↔BE protocol versioning (vNext §11.6)."""
from __future__ import annotations

from app.api import protocol as P


def test_negotiate_caps_at_server_max():
    # A client asking for a newer version than the server speaks → server max.
    assert P.negotiate(99, server_max=3, min_supported=1) == 3


def test_negotiate_honours_a_supported_older_client():
    # One-back: a client on the previous version is served at that version.
    assert P.negotiate(2, server_max=3, min_supported=2) == 2


def test_negotiate_floors_at_min_supported():
    # A too-old client is served the oldest supported (best-effort).
    assert P.negotiate(1, server_max=3, min_supported=2) == 2


def test_negotiate_legacy_client_defaults_to_1():
    assert P.negotiate(None, server_max=1, min_supported=1) == 1
    assert P.negotiate(0, server_max=1, min_supported=1) == 1


def test_is_compatible_window():
    assert P.is_compatible(3, min_supported=2) is True
    assert P.is_compatible(2, min_supported=2) is True
    assert P.is_compatible(1, min_supported=2) is False   # too old
    assert P.is_compatible(None, min_supported=1) is True


def test_known_and_unknown_frames():
    assert P.is_known_frame("token") is True
    assert P.is_known_frame("stage") is True
    assert P.is_known_frame("some_future_frame") is False
    # Unknown frame handling never raises (ignore-with-log).
    P.handle_unknown_frame("some_future_frame")


def test_handshake_info_shape():
    info = P.handshake_info(P.PROTOCOL_VERSION)
    assert info["protocol_version"] == P.PROTOCOL_VERSION
    assert info["server_max"] == P.PROTOCOL_VERSION
    assert info["compatible"] is True
