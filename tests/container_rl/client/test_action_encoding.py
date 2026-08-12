"""Regression tests for the multi-head actions the TUI sends to the server.

Every action head reserves index 0 for no-op, so the env reads each one back
as ``head - 1``.  The TUI used to send plain 0-based game values, which shifted
every action down by one: pressing Produce dispatched to Buy Warehouse, so
money drained away and no containers ever appeared in the factory store.

These tests drive the env with exactly the arrays the TUI builds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from container_rl.client.tui import _mh
from container_rl.env.container import (
    ACTION_BUY_FACTORY,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_MOVE_LOAD,
    ACTION_PASS,
    ACTION_PRODUCE,
    LEAVE_IDLE,
    PURCHASE_STOP,
    ActionEncoder,
    ContainerJaxEnv,
)

jax.config.update("jax_disable_jit", True)


@pytest.fixture
def env():
    e = ContainerJaxEnv(num_players=2, num_colors=5)
    e.reset(seed=1)
    return e


def step(env, arr):
    env.step(jnp.array(arr, dtype=jnp.int32))
    return env.state


def store_total(state, player):
    return int(state.factory_store[player].sum())


def hand_over(env, player):
    """Pass until it is *player*'s go (each player gets two actions a turn)."""
    enc = ActionEncoder(2, 5)
    for _ in range(6):
        if int(env.state.current_player) == player:
            return env.state
        env.step(enc.encode(ACTION_PASS, {}))
    raise AssertionError(f"never reached player {player}")


def test_heads_are_one_indexed():
    """Plain game values go in; the env's no-op offset comes out."""
    assert _mh(ACTION_PRODUCE, color=0, slot=0) == [ACTION_PRODUCE + 1, 0, 1, 1, 0]
    assert _mh(ACTION_PRODUCE) == [ACTION_PRODUCE + 1, 0, 0, 0, 0]
    # The purchase head has its own encoding and is passed through untouched.
    assert _mh(ACTION_MOVE_LOAD, opp=1, purchase=PURCHASE_STOP)[4] == PURCHASE_STOP


def test_produce_fills_the_factory_store(env):
    """The reported bug: produce charged money and produced nothing."""
    enc = ActionEncoder(2, 5)
    env.step(enc.encode(ACTION_BUY_FACTORY, {"color": 1}))  # second factory, $6
    cash_before = int(env.state.cash[0])
    stored_before = store_total(env.state, 0)
    red_at_2_before = int(env.state.factory_store[0, 0, 1])

    state = step(env, _mh(ACTION_PRODUCE))  # opening action: $1 union dues
    assert int(state.produce_active) == 1
    assert int(state.cash[0]) == cash_before - 1

    state = step(env, _mh(ACTION_PRODUCE, color=0, slot=1))  # Red at $2
    state = step(env, _mh(ACTION_PRODUCE, color=1, slot=3))  # Green at $4

    assert int(state.produce_active) == 0
    assert store_total(state, 0) == stored_before + 2
    assert int(state.factory_store[0, 0, 1]) == red_at_2_before + 1  # Red joined the $2 slot
    assert int(state.factory_store[0, 1, 3]) == 1                    # Green at $4
    # Union dues only — producing itself is free, and no warehouse was bought.
    assert int(state.cash[0]) == cash_before - 1
    assert int(state.warehouse_count[0]) == 1


def test_produce_leave_idle_costs_nothing(env):
    stored_before = store_total(env.state, 0)
    step(env, _mh(ACTION_PRODUCE))
    state = step(env, _mh(ACTION_PRODUCE, color=0, slot=LEAVE_IDLE))
    assert int(state.produce_active) == 0
    assert store_total(state, 0) == stored_before


def test_buy_from_factory_lands_in_the_harbour(env):
    """Opening the shop buys nothing; the purchase that follows sets the price."""
    step(env, _mh(ACTION_PRODUCE))
    step(env, _mh(ACTION_PRODUCE, color=0, slot=0))  # P0 stocks Red at $1
    state = hand_over(env, 1)

    state = step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, opp=0, purchase=PURCHASE_STOP))
    assert int(state.shopping_active) == 1
    assert store_total(state, 0) == store_total(env.state, 0)  # nothing bought yet

    seller_cash = int(state.cash[0])
    buyer_cash = int(state.cash[1])
    # purchase=4 asks for a $5 harbour price on the cheapest Red available.
    state = step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, opp=0, color=0, purchase=4))

    assert int(state.harbour_store[1, 0, 4]) == 1   # Red sitting at $5
    assert int(state.cash[1]) == buyer_cash - 1     # paid the $1 asking price
    assert int(state.cash[0]) == seller_cash + 1


def test_move_load_puts_cargo_on_the_ship(env):
    step(env, _mh(ACTION_PRODUCE))
    step(env, _mh(ACTION_PRODUCE, color=0, slot=0))
    hand_over(env, 1)
    step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, opp=0, purchase=PURCHASE_STOP))
    step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, opp=0, color=0, purchase=4))
    if int(env.state.shopping_active):
        step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, purchase=PURCHASE_STOP))
    state = hand_over(env, 0)

    state = step(env, _mh(ACTION_MOVE_LOAD, opp=0, purchase=PURCHASE_STOP))
    assert int(state.shopping_active) == 1

    buyer_cash = int(state.cash[0])
    state = step(env, _mh(ACTION_MOVE_LOAD, opp=0, color=0, purchase=1))
    assert int(state.ship_contents[0, 0]) == 1      # colour 0 stored as colour+1
    assert int(state.cash[0]) == buyer_cash - 5     # the $5 the seller asked


def test_stop_ends_shopping(env):
    step(env, _mh(ACTION_PRODUCE))
    step(env, _mh(ACTION_PRODUCE, color=0, slot=0))
    hand_over(env, 1)
    step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, opp=0, purchase=PURCHASE_STOP))
    state = step(env, _mh(ACTION_BUY_FROM_FACTORY_STORE, purchase=PURCHASE_STOP))
    assert int(state.shopping_active) == 0
