"""Regression tests for the ``get_state`` handler used when joining a game.

Joining an existing game is the one place a client blocks on a single reply.
When that reply never came the TUI waited out its timeout and then unwound the
whole app -- over ssh that ended the session, so picking a game looked like
being logged out.  These tests pin the handler's side of that contract.
"""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest

from container_rl.server import server as server_mod
from container_rl.server.server import ClientHandler


class FakeSocket:
    """Collects the frames the handler writes."""

    def __init__(self):
        self.frames: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.frames.append(data)

    def close(self) -> None:
        pass


def sent_messages(sock: FakeSocket) -> list[dict]:
    """Decode every length-prefixed JSON frame the handler sent."""
    buf = b"".join(sock.frames)
    out = []
    while len(buf) >= 4:
        (n,) = struct.unpack(">I", buf[:4])
        out.append(json.loads(buf[4 : 4 + n]))
        buf = buf[4 + n :]
    return out


def make_handler(manager) -> tuple[ClientHandler, FakeSocket]:
    sock = FakeSocket()
    fake_server = SimpleNamespace(manager=manager)
    handler = ClientHandler(sock, ("127.0.0.1", 1234), fake_server)
    handler.game_id = 7
    handler.player_index = 0
    return handler, sock


def fake_state():
    return SimpleNamespace(
        current_player=1, actions_taken=0, auction_active=0,
        produce_active=0, shopping_active=0, game_over=0,
    )


@pytest.fixture
def stub_serialisers(monkeypatch):
    """Keep the tests off the real env/pickle machinery."""
    monkeypatch.setattr(
        "container_rl.server.protocol.serialize_state", lambda state: b"\xde\xad"
    )
    monkeypatch.setattr(
        server_mod, "_state_to_json_data", lambda state: {"cash": [10, 20]}
    )


def test_unloadable_state_reports_an_error(stub_serialisers):
    """A game whose state cannot be rebuilt must say so straight away.

    The client waits on this reply; silence is what made the TUI time out and
    drop the ssh session instead of showing the reason.
    """
    def boom(game_id):
        raise ValueError(
            "Saved game state is from an incompatible older format; missing "
            "field(s): secret_card_values. Start a new game."
        )

    handler, sock = make_handler(SimpleNamespace(get_state=boom))
    handler._handle_get_state()

    msgs = sent_messages(sock)
    assert [m["type"] for m in msgs] == ["error"]
    assert "secret_card_values" in msgs[0]["payload"]["message"]


def test_successful_get_state_includes_decoded_state_data(stub_serialisers):
    """``state_update`` carries the decoded fields, like the action broadcast."""
    manager = SimpleNamespace(
        get_state=lambda game_id: fake_state(),
        play_ai_turn_if_needed=lambda game_id: None,
    )
    handler, sock = make_handler(manager)
    handler._handle_get_state()

    msgs = sent_messages(sock)
    assert [m["type"] for m in msgs] == ["state_update"]
    payload = msgs[0]["payload"]
    assert payload["state"] == "dead"
    assert payload["state_data"] == {"cash": [10, 20]}
    assert payload["current_player"] == 1


def test_ai_failure_does_not_follow_a_good_state_with_an_error(stub_serialisers):
    """An AI auto-play crash must not be reported as a failed get_state.

    The client has already been served correctly at that point; a trailing
    ``error`` would blame the join for something that happened afterwards.
    """
    def boom(game_id):
        raise RuntimeError("model exploded")

    manager = SimpleNamespace(
        get_state=lambda game_id: fake_state(),
        play_ai_turn_if_needed=boom,
    )
    handler, sock = make_handler(manager)
    handler._handle_get_state()

    assert [m["type"] for m in sent_messages(sock)] == ["state_update"]


def test_get_state_without_a_game_is_rejected():
    handler, sock = make_handler(SimpleNamespace())
    handler.game_id = None
    handler._handle_get_state()

    msgs = sent_messages(sock)
    assert msgs[0]["type"] == "error"
    assert msgs[0]["payload"]["message"] == "Not in a game."
