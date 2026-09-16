"""Regression tests for the end of the game in the networked client.

When a game finished, players were left looking at a board that said
"Waiting for <name> to play…" for good.  The waiting loop only let go when the
turn came back round to them or an auction opened, and a finished game does
neither -- so unless the very last move was your own, the final scores were
never shown and nothing said the game was over.

These tests pin down the replacement: every route into a finished game lands
on the end-of-game screen, that screen only leaves on a deliberate key, and the
scores it shows are the scores the game is actually decided on.
"""

from __future__ import annotations

import contextlib
import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from rich.console import Console

from container_rl.client import tui
from container_rl.env.container import ContainerFunctional, ContainerJaxEnv, ContainerParams
from container_rl.server.protocol import serialize_state

jax.config.update("jax_disable_jit", True)

NC = 5
ESC = "\x1b"
LEFT = "\x1b[D"


def _i32(x):
    return jnp.array(x, dtype=jnp.int32)


@pytest.fixture
def state():
    params = ContainerParams(num_players=3, num_colors=NC)
    return ContainerFunctional().initial(jax.random.PRNGKey(0), params)


@pytest.fixture(autouse=True)
def names(monkeypatch):
    monkeypatch.setattr(tui, "PLAYER_NAMES", {i: f"Player {i+1}" for i in range(5)})


def _finished(s, supply=(0, 0, 3, 3, 3)):
    return s._replace(game_over=_i32(1), container_supply=_i32(list(supply)))


# ── the score has to be the real score ────────────────────────────────────

@pytest.mark.parametrize("seed", range(40))
def test_breakdown_matches_the_env_net_worth(state, seed):
    """Harbour at $2, ship at $3, the big island colour discarded, loans -$11."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    s = state._replace(
        cash=jax.random.randint(k1, (3,), 0, 60).astype(jnp.int32),
        loans=jax.random.randint(k2, (3,), 0, 3).astype(jnp.int32),
        harbour_store=jax.random.randint(k3, state.harbour_store.shape, 0, 2).astype(jnp.int32),
        ship_contents=jax.random.randint(k4, state.ship_contents.shape, 0, NC + 1).astype(jnp.int32),
        # Small counts so ties -- including ties with the 5/10 colour -- are common.
        island_store=jax.random.randint(k5, state.island_store.shape, 0, 3).astype(jnp.int32),
    )
    func = ContainerFunctional()
    for p in range(3):
        assert tui._net_worth(s, p, NC) == int(func._net_worth(s, p, NC)), p


def test_a_tie_with_the_five_ten_colour_discards_that_colour(state):
    cards = state.secret_card_values
    five_ten = int(jnp.argmax(cards[0] == -1))
    other = (five_ten + 1) % NC
    island = jnp.zeros(NC, dtype=jnp.int32).at[five_ten].set(2).at[other].set(2)
    s = state._replace(island_store=state.island_store.at[0].set(island))
    bd = tui._score_breakdown(s, 0, NC)
    assert bd["discarded"] == five_ten
    assert bd["island"] == 2 * int(cards[0, other])


def test_equal_scores_share_a_position(state):
    s = state._replace(cash=_i32([20, 30, 20]))
    s = s._replace(island_store=jnp.zeros_like(s.island_store),
                   harbour_store=jnp.zeros_like(s.harbour_store),
                   ship_contents=jnp.zeros_like(s.ship_contents),
                   loans=_i32([0, 0, 0]))
    assert [(pos, p) for pos, p, _ in tui._standings(s, NC, 3)] == [(1, 1), (2, 0), (2, 2)]


# ── the screen ────────────────────────────────────────────────────────────

def _text(renderable, width=100, height=40):
    console = Console(no_color=True, width=width, height=height, force_terminal=False)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_screen_names_the_winner_and_your_finish(state, monkeypatch):
    s = _finished(state._replace(cash=_i32([5, 50, 10])))
    out = _text(tui._render_game_over(s, NC, 3, my_player=2))
    assert "GAME OVER" in out
    assert "Player 2 wins" in out
    assert "You finished 2nd of 3" in out
    assert "Final standings" in out and "Player statistics" in out
    assert "containers ran out" in out
    assert "[/" not in out, "markup leaked onto the screen"


def test_screen_reveals_every_secret_card(state):
    s = _finished(state)
    out = _text(tui._render_game_over(s, NC, 3, my_player=0))
    assert out.count("5/10") == 3


def test_screen_reports_a_shared_win(state):
    s = _finished(state._replace(cash=_i32([40, 40, 10]),
                                 island_store=jnp.zeros_like(state.island_store),
                                 harbour_store=jnp.zeros_like(state.harbour_store)))
    out = _text(tui._render_game_over(s, NC, 3, my_player=0))
    assert "Tie for first" in out
    assert "(shared)" in out


def test_screen_fits_five_players_on_a_small_terminal():
    """ssh terminals are often 80 columns: the render must not raise."""
    params = ContainerParams(num_players=5, num_colors=NC)
    s = _finished(ContainerFunctional().initial(jax.random.PRNGKey(3), params))
    out = _text(tui._render_game_over(s, NC, 5, my_player=4), width=80, height=30)
    for p in range(1, 6):
        assert f"Player {p}" in out


def test_screen_says_when_the_move_limit_ended_it(state):
    s = state._replace(game_over=_i32(1))
    assert "move limit" in _text(tui._render_game_over(s, NC, 3))


# ── getting there, and getting out ────────────────────────────────────────

@pytest.fixture
def table(monkeypatch):
    """A 3-player game seen from seat 0, served by a scripted fake server."""
    env = ContainerJaxEnv(num_players=3, num_colors=NC)
    env.reset(seed=1)
    queue: list[dict] = []
    game = SimpleNamespace(env=env, on_poll=lambda n: None, screens=[])

    def push():
        queue.append({"type": "state_update",
                      "payload": {"state": serialize_state(env.state).hex(),
                                  "game_over": int(env.state.game_over)}})

    game.push = push

    class FakeClient:
        sock = object()

        def send(self, msg_type, payload=None):
            if msg_type == "get_state":
                push()

        def disconnect(self):
            pass

    polls = itertools.count()

    def drain():
        n = next(polls)
        assert n < 400, "the client never reached the end-of-game screen"
        game.on_poll(n)
        out = list(queue)
        queue.clear()
        return out

    class FakeLive:
        def update(self, r):
            game.screens.append(r)

        def refresh(self):
            pass

    @contextlib.contextmanager
    def live(_renderable):
        yield FakeLive()

    real_render = tui._render_game_over

    def spy(*a, **k):
        game.game_over_shown = True
        return real_render(*a, **k)

    game.game_over_shown = False
    monkeypatch.setattr(tui, "_render_game_over", spy)
    monkeypatch.setattr(tui, "CLIENT", FakeClient())
    monkeypatch.setattr(tui, "PLAYER_INDEX", 0)
    monkeypatch.setattr(tui, "NUM_PLAYERS", 3)
    monkeypatch.setattr(tui, "NUM_COLORS", NC)
    monkeypatch.setattr(tui, "_drain_server", drain)
    monkeypatch.setattr(tui, "_game_live", live)
    monkeypatch.setattr(tui._time, "sleep", lambda *a: None)
    return game


def _keys(monkeypatch, until, keys):
    """Feed nothing until ``until()`` holds, then feed ``keys`` in order."""
    pending = list(keys)

    def _key(timeout=None):
        if not until() or not pending:
            return ""
        return pending.pop(0)

    monkeypatch.setattr(tui, "_key", _key)
    return pending


def test_game_ending_on_someone_elses_move_shows_the_end_screen(table, monkeypatch):
    """The hang itself: we are waiting on seat 1 when seat 1 ends the game."""
    env = table.env
    env.state = env.state._replace(current_player=_i32(1))

    def seat_one_finishes(n):
        if n == 5:
            env.state = _finished(env.state)
            table.push()

    table.on_poll = seat_one_finishes
    _keys(monkeypatch, lambda: table.game_over_shown, ["q"])
    assert tui._gameplay() is None
    assert table.game_over_shown


def test_rejoining_a_finished_game_opens_on_the_end_screen(table, monkeypatch):
    table.env.state = _finished(table.env.state._replace(current_player=_i32(2)))
    _keys(monkeypatch, lambda: table.game_over_shown, [ESC])
    assert tui._gameplay() is tui.BACK
    assert table.game_over_shown


def test_stray_keys_do_not_leave_the_end_screen(table, monkeypatch):
    """Space was "pass" a moment ago; it must not drop the ssh session."""
    table.env.state = _finished(table.env.state)
    pending = _keys(monkeypatch, lambda: table.game_over_shown, [" ", "1", "\r", "x", "q"])
    assert tui._gameplay() is None
    assert pending == [], "left the end screen before q was pressed"


def test_history_is_reachable_from_the_end_screen_and_back(table, monkeypatch):
    env = table.env
    env.state = env.state._replace(current_player=_i32(1))

    def moves_then_finish(n):
        if n == 2:
            env.state = env.state._replace(cash=env.state.cash.at[1].add(1))
            table.push()
        if n == 4:
            env.state = _finished(env.state)
            table.push()

    table.on_poll = moves_then_finish
    seen = []
    real = tui._render

    def render_spy(*a, **k):
        seen.append(k.get("hist_msg", ""))
        return real(*a, **k)

    monkeypatch.setattr(tui, "_render", render_spy)
    _keys(monkeypatch, lambda: table.game_over_shown, [LEFT, "\x1b[C", "q"])
    assert tui._gameplay() is None
    assert any(seen), "← on the end screen did not open the move history"
