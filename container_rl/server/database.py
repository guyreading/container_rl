"""SQLite persistence for games, players, and game state.

Tables
------
players
    id, name, password_hash, created_at

games
    id, code (unique), status (lobby|active|finished),
    num_players, num_colors, seed, created_at, finished_at

game_players
    game_id, player_index, player_id, joined_at

game_states
    game_id, state_blob, step_count, saved_at
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
import string
import time
from typing import Any, Optional

import numpy as np

WORDS = [
    "RED", "BLUE", "GREEN", "GOLD", "SILVER", "IRON", "COPPER",
    "FALCON", "WOLF", "BEAR", "HAWK", "LYNX", "TIGER", "EAGLE",
    "RIVER", "STONE", "CLOUD", "STORM", "FLAME", "OCEAN",
]


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _generate_code() -> str:
    import random as _random
    word = _random.choice(WORDS)
    num = _random.randint(10, 99)
    return f"{word}-{num}"


class Database:
    def __init__(self, path: str = "container_server.db"):
        self.path = path
        self._init_db()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    password_hash TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'lobby',
                    num_players INTEGER NOT NULL,
                    num_colors INTEGER NOT NULL DEFAULT 5,
                    seed INTEGER NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS game_players (
                    game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
                    player_index INTEGER NOT NULL,
                    player_id INTEGER REFERENCES players(id),
                    joined_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (game_id, player_index)
                );

                CREATE TABLE IF NOT EXISTS game_states (
                    game_id INTEGER PRIMARY KEY REFERENCES games(id) ON DELETE CASCADE,
                    state_blob BLOB NOT NULL,
                    step_count INTEGER DEFAULT 0,
                    saved_at TEXT DEFAULT (datetime('now'))
                );
            """)

    # ------------------------------------------------------------------
    # players
    # ------------------------------------------------------------------

    def upsert_player(self, name: str, password: str | None = None) -> int:
        """Find or create a player.  Returns the player id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM players WHERE name = ?", (name,)
            ).fetchone()
            if row:
                pw_hash = _hash_password(password) if password else None
                if row["password_hash"] and row["password_hash"] != pw_hash:
                    raise ValueError(f"Player '{name}' exists with a different password.")
                if pw_hash and not row["password_hash"]:
                    conn.execute(
                        "UPDATE players SET password_hash = ? WHERE id = ?",
                        (pw_hash, row["id"]),
                    )
                return row["id"]
            pw_hash = _hash_password(password) if password else None
            cur = conn.execute(
                "INSERT INTO players (name, password_hash) VALUES (?, ?)",
                (name, pw_hash),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # games
    # ------------------------------------------------------------------

    def create_game(
        self, num_players: int, num_colors: int = 5, seed: int | None = None,
    ) -> tuple[int, str]:
        """Create a new game.  Returns (game_id, game_code)."""
        code = _generate_code()
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO games (code, num_players, num_colors, seed)
                   VALUES (?, ?, ?, ?)""",
                (code, num_players, num_colors, seed),
            )
            game_id = cur.lastrowid
        return game_id, code

    def get_game_by_code(self, code: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM games WHERE code = ?", (code,)).fetchone()
            return dict(row) if row else None

    def get_game_by_id(self, game_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
            return dict(row) if row else None

    def list_joinable_games(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT g.*, COUNT(gp.player_index) as slots_filled
                FROM games g
                LEFT JOIN game_players gp ON g.id = gp.game_id
                WHERE g.status = 'lobby'
                GROUP BY g.id
                ORDER BY g.created_at DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def set_game_status(self, game_id: int, status: str) -> None:
        with self._connect() as conn:
            extra = ", finished_at = datetime('now')" if status == "finished" else ""
            conn.execute(
                f"UPDATE games SET status = ?{extra} WHERE id = ?",
                (status, game_id),
            )

    def count_game_players(self, game_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM game_players WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            return row["cnt"]

    # ------------------------------------------------------------------
    # game_players (slots)
    # ------------------------------------------------------------------

    def assign_player_slot(
        self, game_id: int, player_id: int, player_index: int | None = None,
    ) -> int:
        """Assign a player to an available slot.  Returns the player_index."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT player_index, player_id FROM game_players WHERE game_id = ? AND player_id = ?",
                (game_id, player_id),
            ).fetchone()
            if existing:
                return existing["player_index"]

            if player_index is not None:
                existing_slot = conn.execute(
                    "SELECT player_id FROM game_players WHERE game_id = ? AND player_index = ?",
                    (game_id, player_index),
                ).fetchone()
                if existing_slot:
                    raise ValueError(f"Slot {player_index} already taken.")
                conn.execute(
                    "INSERT OR REPLACE INTO game_players (game_id, player_index, player_id) VALUES (?, ?, ?)",
                    (game_id, player_index, player_id),
                )
                return player_index

            taken = set(
                r["player_index"]
                for r in conn.execute(
                    "SELECT player_index FROM game_players WHERE game_id = ?",
                    (game_id,),
                ).fetchall()
            )
            game = conn.execute("SELECT num_players FROM games WHERE id = ?", (game_id,)).fetchone()
            for i in range(game["num_players"]):
                if i not in taken:
                    conn.execute(
                        "INSERT INTO game_players (game_id, player_index, player_id) VALUES (?, ?, ?)",
                        (game_id, i, player_id),
                    )
                    return i
            raise ValueError("Game is full.")

    def get_game_players(self, game_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT gp.player_index, p.name, p.id as player_id
                FROM game_players gp
                JOIN players p ON gp.player_id = p.id
                WHERE gp.game_id = ?
                ORDER BY gp.player_index
            """, (game_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_player_slot(self, game_id: int, player_id: int) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT player_index FROM game_players WHERE game_id = ? AND player_id = ?",
                (game_id, player_id),
            ).fetchone()
            return row["player_index"] if row else None

    # ------------------------------------------------------------------
    # game state persistence
    # ------------------------------------------------------------------

    def save_state(self, game_id: int, state_blob: bytes, step_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO game_states (game_id, state_blob, step_count)
                   VALUES (?, ?, ?)""",
                (game_id, state_blob, step_count),
            )

    def load_state(self, game_id: int) -> bytes | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_blob FROM game_states WHERE game_id = ?", (game_id,),
            ).fetchone()
            return row["state_blob"] if row else None

    def load_step_count(self, game_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT step_count FROM game_states WHERE game_id = ?", (game_id,),
            ).fetchone()
            return row["step_count"] if row else 0
