"""Game codes must stay unique without failing game creation.

Regression: codes were WORD-NN from 20 words (1,800 total), picked at random
with no retry, so ``create_game`` hit "UNIQUE constraint failed: games.code"
after only a few dozen games.
"""

from __future__ import annotations

import re

import pytest

from container_rl.server import database
from container_rl.server.database import Database

OLD_CODE_SPACE = 20 * 90


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "codes.db"))


def test_code_space_is_at_least_100x_the_old_one():
    assert database.CODE_SPACE >= 100 * OLD_CODE_SPACE
    assert len(set(database.WORDS)) == len(database.WORDS)


def test_generated_codes_are_word_dash_number():
    for _ in range(200):
        code = database._generate_code()
        word, num = code.split("-")
        assert word in database.WORDS
        assert re.fullmatch(r"\d{4}", num)
        assert code == code.upper()


def test_many_games_get_unique_codes(db):
    codes = [db.create_game(num_players=2)[1] for _ in range(500)]
    assert len(set(codes)) == len(codes)


def test_create_game_retries_when_code_is_taken(db, monkeypatch):
    _, first = db.create_game(num_players=2)
    draws = iter([first, first, "WOLF-1234"])
    monkeypatch.setattr(database, "_generate_code", lambda: next(draws))

    game_id, code = db.create_game(num_players=2)

    assert code == "WOLF-1234"
    assert db.get_game_by_code(code)["id"] == game_id


def test_create_game_gives_up_after_max_attempts(db, monkeypatch):
    _, first = db.create_game(num_players=2)
    monkeypatch.setattr(database, "_generate_code", lambda: first)

    with pytest.raises(RuntimeError, match="free game code"):
        db.create_game(num_players=2)
