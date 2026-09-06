"""Regression tests for auctions in games with AI opponents.

Bidding is the one move a player makes out of turn: the seller stays
``current_player`` for the whole auction, and everybody else answers while
they wait.  The AI loop is driven entirely by whose turn it is, so an auction
opened by a human was never put to the AI seats — no bids arrived, the round
never advanced, and the seller's client sat waiting for a state that could
not come.  That branch of the TUI reads no keys, and in raw mode Ctrl-C is
just another byte, so the game did not merely pause: the player could not
quit out of it either.

The second thing pinned here is who may bid.  The bidder is named by a raw
player index on the opponent head, and nothing in the env ties that to the
seat that sent the action, so an AI turn taken during an open auction can
enter a bid on a human's behalf and spend their cash.  The AI must bid for
its own seats and no others.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from container_rl.env.container import (
    ACTION_MOVE_AUCTION,
    LOCATION_OPEN_SEA,
    ActionEncoder,
    ContainerJaxEnv,
)
from container_rl.server import game_manager
from container_rl.server.game_manager import GameManager

jax.config.update("jax_disable_jit", True)

GAME_ID = 1
NC = 5


class StubDB:
    """The three calls ``process_action`` makes on the database."""

    def __init__(self, np_):
        self.np_ = np_
        self.status = None

    def get_game_players(self, game_id):
        return [{"player_index": i, "name": f"Player {i+1}", "is_ai": i > 0}
                for i in range(self.np_)]

    def save_state(self, game_id, blob, step_count):
        pass

    def set_game_status(self, game_id, status):
        self.status = status


@pytest.fixture
def game():
    """A game where seat 0 is human, the rest are AI, and seat 0 can auction.

    Returns ``(manager, env)``.  No model is configured, so the AI plays the
    masked-random fallback — the same path a missing or broken checkpoint
    takes in production.
    """
    def _make(np_=3, seller=0):
        env = ContainerJaxEnv(num_players=np_, num_colors=NC)
        env.reset(seed=1)
        s = env.state
        env.state = s._replace(
            current_player=jnp.array(seller, dtype=s.current_player.dtype),
            ship_contents=s.ship_contents.at[seller].set(
                jnp.array([1, 2, 3, 0, 0], dtype=s.ship_contents.dtype)),
            ship_location=s.ship_location.at[seller].set(
                jnp.array(LOCATION_OPEN_SEA, dtype=s.ship_location.dtype)),
        )
        sent = []
        db = StubDB(np_)
        mgr = GameManager(db, lambda gid, t, p: sent.append((t, p)))
        mgr._envs[GAME_ID] = env
        mgr._encoders[GAME_ID] = ActionEncoder(np_, NC)
        mgr._ai_slots[GAME_ID] = [p for p in range(np_) if p != 0]
        return mgr, env, sent

    return _make


def _open_auction(mgr, env, seller):
    """Have *seller* play action 7."""
    enc = mgr.get_encoder(GAME_ID)
    return mgr.process_action(GAME_ID, seller, enc.encode(ACTION_MOVE_AUCTION, {}))


@pytest.mark.parametrize("np_", [3, 4, 5])
def test_ai_bids_when_a_human_opens_an_auction(game, np_):
    """The auction must not be left open with the AI seats yet to bid."""
    mgr, env, _sent = game(np_)
    _open_auction(mgr, env, seller=0)

    bids = env.state.auction_bids.tolist()
    assert all(b >= 0 for b in bids), f"AI seats never bid: {bids}"
    # Every bid is in, so it is the seller's decision now — or the auction has
    # already resolved.  Either way the game is not waiting on the AI.
    if int(env.state.auction_active):
        assert int(env.state.auction_round) == 1


def test_ai_bids_are_affordable(game):
    """A bid above the bidder's cash is not a bid the env can honour."""
    mgr, env, _sent = game(4)
    cash_before = env.state.cash.tolist()
    _open_auction(mgr, env, seller=0)
    for p, bid in enumerate(env.state.auction_bids.tolist()):
        assert bid <= cash_before[p], f"seat {p} bid {bid} with ${cash_before[p]}"


def test_the_bids_reach_the_players(game):
    """Clients only redraw on a state_update, so the bids have to be sent."""
    mgr, env, sent = game(3)
    _open_auction(mgr, env, seller=0)
    states = [p for t, p in sent if t == "state_update"]
    assert states, "no state broadcast at all"
    assert int(states[-1]["auction_active"]) == 0 or \
        int(env.state.auction_round) == 1


def test_ai_does_not_bid_for_a_human(game):
    """An AI seller must leave the human bidders alone.

    The bidder head is a raw player index and the env does not check it
    against the seat that sent the action, so an unlucky AI action can enter
    a bid for somebody else and take the money out of their pocket.
    """
    mgr, env, _sent = game(3, seller=1)  # seat 1 is AI and holds the cargo
    mgr._ai_slots[GAME_ID] = [1, 2]
    enc = mgr.get_encoder(GAME_ID)
    cash_before = env.state.cash.tolist()

    mgr.process_action(GAME_ID, 1, enc.encode(ACTION_MOVE_AUCTION, {}))

    st = env.state
    # The human has not answered, so the auction cannot have moved on.
    assert int(st.auction_active) == 1, "the auction closed without the human"
    assert int(st.auction_round) == 0
    assert int(st.auction_bids[0]) < 0, "the human's bid was entered for them"
    assert st.cash.tolist()[0] == cash_before[0], "the human was charged for a bid"


def test_a_player_cannot_bid_in_someone_elses_name(game):
    """The bidder head is a raw index, so the door has to be watched."""
    mgr, env, _sent = game(3)
    _open_auction(mgr, env, seller=0)
    if int(env.state.auction_active) == 0:
        pytest.skip("auction already resolved; nothing left to bid on")

    cash_before = env.state.cash.tolist()
    result = mgr.process_action(GAME_ID, 2, [ACTION_MOVE_AUCTION + 1, 1, 0, 0, 5])

    assert result.get("error"), "the bid was accepted in another seat's name"
    assert env.state.cash.tolist() == cash_before


def test_an_all_ai_auction_finishes(game, monkeypatch):
    """With nobody human left to ask, the auction must run to the end.

    The turn budget is pinned to one so the assertion is about the auction
    that was just opened: an all-AI table otherwise plays on for two hundred
    turns and lands wherever it lands.
    """
    mgr, env, _sent = game(3, seller=1)
    mgr._ai_slots[GAME_ID] = [0, 1, 2]
    monkeypatch.setattr(game_manager, "MAX_AI_TURN_STEPS", 1)
    enc = mgr.get_encoder(GAME_ID)
    mgr.process_action(GAME_ID, 1, enc.encode(ACTION_MOVE_AUCTION, {}))
    assert int(env.state.auction_active) == 0, "the auction never closed"
