"""Regression tests for keystrokes that used to kill the TUI outright.

Pressing ``4`` (Buy from Factory) opened the opponent picker, and picking an
opponent is a single-keypress menu: with three players it offers two options.
That menu returned whatever digit was pressed without checking it named an
option, so a stray ``4`` came back as 4 and the ``cand[ch-1]`` that followed
raised IndexError.  Nothing caught it, so it unwound out of ``main`` and ended
the process — and because the TUI is people's ssh login shell, that shows up
as being dropped off the server mid-game, with the traceback going to a log
file rather than their terminal.

Two things are locked down here:

* every menu answer is either None or names a real option, whatever is typed;
* a crash inside an action is contained, so the game survives one.

The two-player case never hit this because the opponent picker short-circuits
when there is only one opponent to pick — which is why it surfaced only once
games moved to a 3-5 player range.
"""

from __future__ import annotations

import itertools
import random

import jax
import jax.numpy as jnp
import pytest

from container_rl.client import tui
from container_rl.env.container import (
    ACTION_PASS,
    ACTION_PRODUCE,
    ActionEncoder,
    ContainerJaxEnv,
)

jax.config.update("jax_disable_jit", True)

NC = 5
ESC = "\x1b"
SEAT = 1  # the seat we play from in these tests


class FakeLive:
    """``rich``'s Live, minus the terminal."""

    def update(self, *a, **k):
        pass

    def refresh(self):
        pass


def _fresh_board(np_):
    """A board where seat 0 has stock to sell and it is seat 1's go."""
    env = ContainerJaxEnv(num_players=np_, num_colors=NC)
    env.reset(seed=1)
    enc = ActionEncoder(np_, NC)
    while int(env.state.current_player) != 0:
        env.step(enc.encode(ACTION_PASS, {}))
    # Seat 0 produces into its factory store at $1 so there is something to buy.
    env.step(jnp.array(tui._mh(ACTION_PRODUCE), dtype=jnp.int32))
    for _ in range(NC + 1):
        if not int(env.state.produce_active):
            break
        pending = [c for c in range(NC) if int(env.state.produce_pending[c]) > 0]
        env.step(jnp.array(
            tui._mh(ACTION_PRODUCE, color=pending[0] if pending else 0, slot=0),
            dtype=jnp.int32))
    while int(env.state.current_player) != SEAT:
        env.step(enc.encode(ACTION_PASS, {}))
    return env


# Building a board runs a few hundred jitless env steps, so do it once per
# player count and rewind the env between tests.
_BOARDS: dict[int, tuple] = {}


@pytest.fixture
def board(monkeypatch):
    """Wire the TUI's globals up to a real env we can step.

    Returns ``(env, sent)``: *sent* collects every action the TUI sends.
    """
    def _make(np_=3):
        if np_ not in _BOARDS:
            env = _fresh_board(np_)
            _BOARDS[np_] = (env, env.state)
        env, snapshot = _BOARDS[np_]
        env.state = snapshot  # rewind: previous tests may have moved it on

        sent, queue = [], []

        class FakeClient:
            sock = object()

            def send(self, msg_type, payload):
                # Submenus send multi-head arrays; the simple actions send a
                # flat index.  The env takes either.
                action = payload["action"] if msg_type == "action_multi" else payload["action_idx"]
                sent.append(action)
                env.step(jnp.array(action, dtype=jnp.int32))
                queue.append({"type": "state_update", "payload": {"state": "00"}})

        monkeypatch.setattr(tui, "STATE", env.state)
        monkeypatch.setattr(tui, "PLAYER_INDEX", SEAT)
        monkeypatch.setattr(tui, "NUM_PLAYERS", np_)
        monkeypatch.setattr(tui, "NUM_COLORS", NC)
        monkeypatch.setattr(tui, "PLAYER_NAMES", {i: f"Player {i+1}" for i in range(np_)})
        monkeypatch.setattr(tui, "CLIENT", FakeClient())
        monkeypatch.setattr(tui, "_drain_server", lambda: [queue.pop(0) for _ in range(len(queue))])
        monkeypatch.setattr(tui, "deserialize_state", lambda _blob: env.state)
        monkeypatch.setattr(tui._time, "sleep", lambda *a: None)
        tui._reset_history()
        return env, sent

    return _make


@pytest.fixture
def press(monkeypatch):
    """Script the keyboard.  ESC is returned once the script runs out.

    The budget catches the other failure mode a rejected key could cause: a
    menu that loops forever because nothing it is fed makes it exit.
    """
    def _press(seq, budget=200):
        pending = list(seq)
        reads = itertools.count()

        def _key(timeout=None):
            assert next(reads) < budget, "menu never finished"
            return pending.pop(0) if pending else ESC

        monkeypatch.setattr(tui, "_key", _key)

    return _press


# ── the menus themselves ──────────────────────────────────────────────────

@pytest.mark.parametrize("n_options,typed", [
    (2, "4"),   # the reported bug: pressing Buy-from-Factory's own key again
    (2, "3"),   # naming a seat number rather than a list position
    (2, "9"),
    (3, "0"),   # the list is 1-indexed, so 0 names nothing
    (5, "7"),
    (9, "0"),
])
def test_input_choice_rejects_digits_that_name_no_option(board, press, n_options, typed):
    """A digit outside the menu is ignored, not handed back as an index."""
    env, _ = board()
    press([typed])  # then ESC, so the menu ends up cancelled
    got = tui._input_choice(FakeLive(), env.state, NC, tui.NUM_PLAYERS,
                            [f"option {i+1}" for i in range(n_options)])
    assert got is None


@pytest.mark.parametrize("n_options,typed,expected", [
    (2, "1", 1),
    (2, "2", 2),
    (5, "4", 4),
    (9, "9", 9),
])
def test_input_choice_accepts_digits_that_do(board, press, n_options, typed, expected):
    env, _ = board()
    press([typed])
    got = tui._input_choice(FakeLive(), env.state, NC, tui.NUM_PLAYERS,
                            [f"option {i+1}" for i in range(n_options)])
    assert got == expected


def test_input_choice_answer_is_always_a_real_option(board, press):
    """The contract every caller relies on, over every digit and menu size."""
    env, _ = board()
    for n_options in range(1, 10):
        for digit in "0123456789":
            press([digit])
            got = tui._input_choice(FakeLive(), env.state, NC, tui.NUM_PLAYERS,
                                    [f"o{i}" for i in range(n_options)])
            assert got is None or 1 <= got <= n_options, (n_options, digit, got)


@pytest.mark.parametrize("np_", [3, 4, 5])
def test_pick_opponent_survives_a_stray_digit(board, press, np_):
    """The exact crash site: ``cand[ch-1]`` with ch past the end of the list."""
    env, _ = board(np_)
    press(["9"])
    assert tui._pick_opponent(FakeLive(), env.state, NC, np_) is None


@pytest.mark.parametrize("np_", [3, 4, 5])
def test_pick_opponent_still_picks(board, press, np_):
    env, _ = board(np_)
    press(["1"])
    expected = tui._opps(int(env.state.current_player), np_)[0]
    assert tui._pick_opponent(FakeLive(), env.state, NC, np_) == expected


# ── the submenu the bug was reported against ──────────────────────────────

def test_buy_from_factory_stray_key_cancels_instead_of_crashing(board, press):
    """Press 4 to open the menu, press 4 again by mistake: nothing happens."""
    env, sent = board(3)
    press(["4"])
    cancelled = tui._submenu_buy_from_factory(FakeLive(), env.state, NC, 3)
    assert cancelled is True
    assert sent == []  # a cancelled menu must not spend the turn


def test_buy_from_factory_still_buys(board, press):
    """The fix must not cost the working path: opponent 1, colour 1, price 1."""
    env, sent = board(3)
    press(["1", "1", "1"])
    tui._submenu_buy_from_factory(FakeLive(), env.state, NC, 3)
    assert sent, "no action reached the server"
    assert sent[0][0] == 4  # ACTION_BUY_FROM_FACTORY_STORE, +1 for the no-op slot


@pytest.mark.parametrize("menu", [
    "_submenu_buy_from_factory",
    "_submenu_move_load",
    "_submenu_buy_factory",
    "_submenu_produce",
])
def test_no_keystroke_sequence_escapes_a_submenu(board, press, menu):
    """Fuzz: whatever is typed, the menu returns rather than raising."""
    rng = random.Random(20260816)
    keyset = list("0123456789") + ["\r", "\x7f", ESC, "", "q"]
    for _ in range(20):
        np_ = rng.choice([3, 4, 5])
        env, _sent = board(np_)
        seq = [rng.choice(keyset) for _ in range(rng.randint(1, 6))]
        press(seq)
        getattr(tui, menu)(FakeLive(), env.state, NC, np_)  # must not raise


# ── the safety net ────────────────────────────────────────────────────────

def test_a_crashing_action_does_not_take_the_session_down(board, press, monkeypatch):
    """Even an unforeseen bug must cost the turn, not the ssh session."""
    env, _ = board(3)

    def boom(*a, **k):
        raise IndexError("list index out of range")

    monkeypatch.setattr(tui, "_submenu_buy_from_factory", boom)
    press([""])
    shown = []

    class RecordingLive(FakeLive):
        def update(self, *a, **k):
            shown.append(a)

    tui._dispatch_action(RecordingLive(), "4", env.state, ActionEncoder(3, NC))
    assert shown, "the player was told nothing about the failure"


def test_dispatch_action_reports_only_real_failures(board, press):
    """The net must not swallow ordinary play: a normal action still sends."""
    env, sent = board(3)
    press([""])
    tui._dispatch_action(FakeLive(), "0", env.state, ActionEncoder(3, NC))  # Pass
    assert len(sent) == 1
