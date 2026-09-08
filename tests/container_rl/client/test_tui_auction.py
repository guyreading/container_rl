"""Regression tests for the client's side of an auction.

Two ways an auction used to take the game away from the player:

* the seller's client asked the server what its move produced and read only
  the first message of the reply.  ``_drain_server`` empties the socket, so
  every position behind that one was gone -- and each bid is broadcast to
  everyone, so the seller was left looking at the moment the auction opened
  while the game had already moved on;
* the screens with nothing to do slept instead of reading the keyboard.  A
  game that stopped advancing therefore took no keys at all, not even q, and
  in raw mode Ctrl-C is just another byte nobody reads.  A pause you cannot
  leave is not distinguishable, from the player's chair, from a crash.

The first is why the seller stopped seeing the game; the second is why they
could not get out of it.
"""

from __future__ import annotations

import contextlib
import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from container_rl.client import tui
from container_rl.env.container import (
    ACTION_MOVE_AUCTION,
    LOCATION_OPEN_SEA,
    ContainerJaxEnv,
)
from container_rl.server.protocol import serialize_state

jax.config.update("jax_disable_jit", True)

NC = 5
ESC = "\x1b"


class FakeLive:
    def update(self, *a, **k):
        pass

    def refresh(self):
        pass


# ── the dropped half of a reply ───────────────────────────────────────────

@pytest.fixture
def batches(monkeypatch):
    """Feed ``_wait_for_state`` a scripted batch of server messages."""
    def _set(msgs):
        served = [False]

        def _drain():
            if served[0]:
                return []
            served[0] = True
            return msgs

        monkeypatch.setattr(tui, "_drain_server", _drain)
        monkeypatch.setattr(tui, "_push_history", lambda *a, **k: None)
        monkeypatch.setattr(tui, "deserialize_state",
                            lambda blob: SimpleNamespace(tag=blob.decode()))
        monkeypatch.setattr(tui._time, "sleep", lambda *a: None)
        monkeypatch.setattr(tui, "STATE", None)

    return _set


def _state_msg(tag: str) -> dict:
    return {"type": "state_update",
            "payload": {"state": tag.encode().hex(), "auction_active": 1}}


def test_the_newest_position_in_a_batch_wins(batches):
    """Three bids arriving together must leave us on the third, not the first."""
    batches([_state_msg("open"), _state_msg("bid1"), _state_msg("bid2")])
    tui._wait_for_state(FakeLive(), NC, 3, timeout=1.0)
    assert tui.STATE.tag == "bid2"


def test_an_action_result_before_the_state_is_still_kept(batches):
    """The description rides in front of the position it describes."""
    batches([{"type": "action_result", "payload": {"desc": "Auction"}},
             _state_msg("open")])
    tui._reset_history()
    tui._wait_for_state(FakeLive(), NC, 3, timeout=1.0)
    assert tui.STATE.tag == "open"
    assert "Auction" in tui.FEEDBACK


def test_an_unreadable_position_is_reported_not_raised(batches, monkeypatch):
    """A blob we cannot parse must not unwind the client and end the session."""
    batches([_state_msg("open")])
    monkeypatch.setattr(tui, "deserialize_state",
                        lambda blob: (_ for _ in ()).throw(ValueError("old save")))
    tui._wait_for_state(FakeLive(), NC, 3, timeout=1.0)  # must not raise
    assert "old save" in tui.FEEDBACK


# ── the screens that used to take no keys ─────────────────────────────────

@pytest.mark.parametrize("key,expected", [
    ("q", "quit"), ("Q", "quit"), (ESC, "back"), ("\x1b[D", "history"),
    ("", ""), ("5", ""),
])
def test_idle_reports_the_keys_that_matter(monkeypatch, key, expected):
    monkeypatch.setattr(tui, "_key", lambda timeout=None: key)
    assert tui._idle(0.0) == expected


@pytest.fixture
def stalled_game(monkeypatch):
    """A real 3-player game where seat 0 can auction and nobody ever bids.

    That is exactly the position the server left players in: the AI seats
    were never asked for a bid, so the round could not advance.
    """
    env = ContainerJaxEnv(num_players=3, num_colors=NC)
    env.reset(seed=1)
    s = env.state
    env.state = s._replace(
        current_player=jnp.array(0, dtype=s.current_player.dtype),
        ship_contents=s.ship_contents.at[0].set(
            jnp.array([1, 2, 3, 0, 0], dtype=s.ship_contents.dtype)),
        ship_location=s.ship_location.at[0].set(
            jnp.array(LOCATION_OPEN_SEA, dtype=s.ship_location.dtype)),
    )

    queue: list[dict] = []
    game = SimpleNamespace(env=env, sent=[], after_send=lambda: None)

    def push():
        queue.append({"type": "state_update",
                      "payload": {"state": serialize_state(env.state).hex(),
                                  "game_over": 0}})

    game.push = push

    class FakeClient:
        sock = object()

        def send(self, msg_type, payload=None):
            if msg_type == "get_state":
                push()
                return
            action = payload["action"] if msg_type == "action_multi" else \
                tui.ActionEncoder(3, NC).to_multi_head(payload["action_idx"])
            game.sent.append([int(x) for x in action])
            env.step(jnp.array(action, dtype=jnp.int32))
            push()
            game.after_send()

        def disconnect(self):
            pass

    # A hard budget on polling: if the loop wedges it never reads a key
    # again, and without this the test would simply hang instead of failing.
    polls = itertools.count()

    def drain():
        assert next(polls) < 400, "the wait loop stopped reading the keyboard"
        out = list(queue)
        queue.clear()
        return out

    @contextlib.contextmanager
    def live(_renderable):
        yield FakeLive()

    monkeypatch.setattr(tui, "CLIENT", FakeClient())
    monkeypatch.setattr(tui, "PLAYER_INDEX", 0)
    monkeypatch.setattr(tui, "NUM_PLAYERS", 3)
    monkeypatch.setattr(tui, "NUM_COLORS", NC)
    monkeypatch.setattr(tui, "PLAYER_NAMES", {i: f"Player {i+1}" for i in range(3)})
    monkeypatch.setattr(tui, "_drain_server", drain)
    monkeypatch.setattr(tui, "_game_live", live)
    monkeypatch.setattr(tui._time, "sleep", lambda *a: None)
    return game


def _script(monkeypatch, keys):
    pending = list(keys)

    def _key(timeout=None):
        return pending.pop(0) if pending else ""

    monkeypatch.setattr(tui, "_key", _key)
    return pending


@pytest.mark.parametrize("key,expected", [("q", None), (ESC, tui.BACK)])
def test_a_stalled_auction_can_still_be_left(stalled_game, monkeypatch, key, expected):
    """Open an auction nobody answers, then quit.  The key has to land."""
    _script(monkeypatch, ["7", key])
    assert tui._gameplay() is expected
    assert int(stalled_game.env.state.auction_active) == 1, \
        "the auction should still be open"


def test_opening_an_auction_still_works(stalled_game, monkeypatch):
    """The escape hatch must not cost the action itself."""
    env = stalled_game.env
    _script(monkeypatch, ["7", "q"])
    tui._gameplay()
    assert int(env.state.auction_active) == 1
    assert int(env.state.auction_seller) == 0
    assert int(env.state.ship_contents[0].sum()) == 0, "cargo went to the auction"


def test_the_seller_sees_the_bids_land(stalled_game, monkeypatch):
    """The seller must reach the accept/reject prompt once the bids are in.

    Both bids land while we are still waiting on the reply to our own move,
    so all three positions arrive in one batch -- the case whose tail used to
    be thrown away, leaving the seller watching a round that had already
    finished.
    """
    env = stalled_game.env

    def bids_arrive():
        if int(env.state.auction_active) and int(env.state.auction_round) == 0:
            for p, bid in ((1, 2), (2, 3)):  # distinct: no tie-break to reason about
                env.step(jnp.array([ACTION_MOVE_AUCTION + 1, p, 0, 0, bid],
                                   dtype=jnp.int32))
                stalled_game.push()

    stalled_game.after_send = bids_arrive
    cash_before = int(env.state.cash[0])
    _script(monkeypatch, ["7", "1", "q"])  # auction, accept, quit
    tui._gameplay()

    # The opening move carries PURCHASE_STOP on the purchase head; a 1 there
    # is the seller saying "accept", which only the decision prompt sends.
    decisions = [a for a in stalled_game.sent
                 if a[0] == ACTION_MOVE_AUCTION + 1 and a[4] == 1]
    assert decisions, "the seller was never asked to accept or reject"
    assert int(env.state.auction_active) == 0, "the auction never resolved"
    assert int(env.state.cash[0]) > cash_before, "the accepted bid was not paid"
