"""Tests for the container-supply slider on the create-game screen.

The screen offers three sliders now: players, AI opponents, and the starting
container supply per colour.  The supply one is the fiddly one -- it has to
follow the rules default (4 per player) while the player count is still being
chosen, but stop following once the player has expressed a preference of their
own, and it must never walk off either end of the allowed range.
"""

from __future__ import annotations

import itertools

import pytest

from container_rl.client import tui
from container_rl.env.container import (
    CONTAINER_SUPPLY_CHOICES,
    default_container_supply,
)

ESC = "\x1b"
UP = "\x1b[A"
DOWN = "\x1b[B"
LEFT = "\x1b[D"
RIGHT = "\x1b[C"
ENTER = "\r"

# Row order on the screen: players, AI opponents, containers.
TO_CONTAINERS = [DOWN, DOWN]
TO_PLAYERS = [UP, UP]


@pytest.fixture
def press(monkeypatch):
    """Script the keyboard; ESC once the script runs out, so a stuck screen fails."""
    def _press(seq, budget=200):
        pending = list(seq)
        reads = itertools.count()

        def _key(timeout=None):
            assert next(reads) < budget, "create screen never finished"
            return pending.pop(0) if pending else ESC

        monkeypatch.setattr(tui, "_key", _key)

    return _press


@pytest.mark.parametrize("extra_players,expected", [
    (0, 12),  # 3 players
    (1, 16),  # 4 players
    (2, 20),  # 5 players
])
def test_default_supply_follows_player_count(press, extra_players, expected):
    """Untouched, the supply bar shows the rules default of 4 per player."""
    press([RIGHT] * extra_players + [ENTER])
    cfg = tui._create_screen()
    assert cfg["num_players"] == 3 + extra_players
    assert cfg["containers_per_color"] == expected
    assert expected == default_container_supply(cfg["num_players"])


def test_supply_can_be_set_by_hand(press):
    """Moving the supply bar left picks the next value down the list."""
    press(TO_CONTAINERS + [LEFT, ENTER])
    cfg = tui._create_screen()
    below_default = CONTAINER_SUPPLY_CHOICES[CONTAINER_SUPPLY_CHOICES.index(12) - 1]
    assert cfg["containers_per_color"] == below_default


def test_hand_set_supply_survives_a_player_count_change(press):
    """Once chosen, the supply stays put even if the player count moves after."""
    press(TO_CONTAINERS + [LEFT] + TO_PLAYERS + [RIGHT, ENTER])
    cfg = tui._create_screen()
    assert cfg["num_players"] == 4
    assert cfg["containers_per_color"] == 9  # not re-defaulted to 16


@pytest.mark.parametrize("key,expected", [
    (LEFT, CONTAINER_SUPPLY_CHOICES[0]),
    (RIGHT, CONTAINER_SUPPLY_CHOICES[-1]),
])
def test_supply_stops_at_the_ends_of_the_range(press, key, expected):
    """Holding a direction clamps to the range rather than running off it."""
    press(TO_CONTAINERS + [key] * 12 + [ENTER])
    cfg = tui._create_screen()
    assert cfg["containers_per_color"] == expected


def test_every_default_is_reachable_on_the_slider():
    """Each supported player count's default is one of the slider's stops."""
    for players in (3, 4, 5):
        assert default_container_supply(players) in CONTAINER_SUPPLY_CHOICES


def test_escape_still_cancels(press):
    press(TO_CONTAINERS + [LEFT])  # then ESC
    assert tui._create_screen() is None
