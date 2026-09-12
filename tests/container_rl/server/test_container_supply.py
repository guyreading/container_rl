"""The chosen container supply has to survive the trip to the game state.

The lobby's supply slider is only meaningful if the number reaches the env:
it goes client -> create_game -> games row -> ContainerJaxEnv, and the env is
built from the row (not from the create call) when the game starts.  Games
created before the setting existed carry 0, which has to keep meaning "the
rules default", not "no containers".
"""

from __future__ import annotations

import json
import sqlite3
import struct
from types import SimpleNamespace

import pytest

from container_rl.server.database import Database
from container_rl.server.game_manager import GameManager
from container_rl.server.server import ClientHandler


@pytest.fixture
def manager(tmp_path):
    db = Database(str(tmp_path / "games.db"))
    return GameManager(db, lambda *a, **k: None)


def start(manager, num_players, **kwargs):
    """Create a full AI-filled game and return its starting supply row."""
    res = manager.create_game_trusted(
        "alice", num_players, 5, seed=7, ai_count=num_players - 1, **kwargs
    )
    assert manager.maybe_start_game(res["game_id"])
    return manager.get_state(res["game_id"]).container_supply


@pytest.mark.parametrize("chosen", [4, 6, 8, 9, 12, 15, 16, 20])
def test_chosen_supply_reaches_the_env(manager, chosen):
    supply = start(manager, 3, containers_per_color=chosen)
    assert list(supply) == [chosen] * 5


@pytest.mark.parametrize("num_players,expected", [(3, 12), (4, 16), (5, 20)])
def test_unset_supply_falls_back_to_the_rules_default(manager, num_players, expected):
    supply = start(manager, num_players)
    assert list(supply) == [expected] * 5


def test_legacy_games_row_gets_the_column_and_the_default(tmp_path):
    """A pre-existing database is migrated in place, old games unchanged."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'lobby',
            num_players INTEGER NOT NULL,
            num_colors INTEGER NOT NULL DEFAULT 5,
            seed INTEGER NOT NULL,
            created_at TEXT,
            finished_at TEXT
        );
        INSERT INTO games (code, num_players, num_colors, seed)
        VALUES ('OLD1', 3, 5, 1);
    """)
    conn.commit()
    conn.close()

    db = Database(str(path))
    assert db.get_game_by_code("OLD1")["containers_per_color"] == 0


class FakeSocket:
    def __init__(self):
        self.frames: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.frames.append(data)

    def close(self) -> None:
        pass


def sent_messages(sock: FakeSocket) -> list[dict]:
    buf = b"".join(sock.frames)
    out = []
    while len(buf) >= 4:
        (n,) = struct.unpack(">I", buf[:4])
        out.append(json.loads(buf[4 : 4 + n]))
        buf = buf[4 + n :]
    return out


@pytest.mark.parametrize("bad", [1, 5, 13, 21, -4])
def test_server_rejects_a_supply_off_the_slider(bad):
    """A hand-crafted client cannot ask for a supply the lobby never offers."""
    sock = FakeSocket()
    handler = ClientHandler(sock, ("127.0.0.1", 1234), SimpleNamespace(manager=None))
    handler._handle_create({
        "player_name": "mallory", "num_players": 3, "containers_per_color": bad,
    })
    messages = sent_messages(sock)
    assert [m["type"] for m in messages] == ["error"]
    assert "containers_per_color" in messages[0]["payload"]["message"]
