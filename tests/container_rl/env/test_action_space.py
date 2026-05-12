"""Tests for the Container RL v3 action space.

Covers all 11 action types under the v3 no-op + per-mode masking design.

Action space layout (all values offset +1 for no-op at index 0):
  [action_type (12), opponent (np), colour (nc+1), price_slot (11), purchase (32)]

Purchase head (always 32, independent of player/colour count):
  Index  0: no-op
  Index  1..5:  harbour $2–$6 (factory store shopping continuation)
  Index  6..30: auction bids
  Index 31:     STOP (end shopping / generic sentinel)

Recurrent actions (produce, buy-from-factory-store, move-load) use an
*enter* step in parallel mode (action_type + relevant param heads active,
all others forced no-op), followed by *continuation* steps in sequential
mode (only the continuing heads active, all others forced no‑op).

These tests call the low‑level handler methods directly
(``_action_buy_factory``, ``_shopping_step``, etc.) so that each
sub‑step can be verified independently.  The full ``transition()``
pipeline is verified in the integration tests.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random

jax.config.update("jax_disable_jit", True)

from container_rl.env.container import (
    ACTION_BUY_FACTORY,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_BUY_WAREHOUSE,
    ACTION_DOMESTIC_SALE,
    ACTION_MOVE_AUCTION,
    ACTION_MOVE_LOAD,
    ACTION_MOVE_SEA,
    ACTION_PASS,
    ACTION_PRODUCE,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    HEAD_ACTION_TYPE,
    HEAD_COLOR,
    HEAD_OPPONENT,
    HEAD_PRICE_SLOT,
    HEAD_PURCHASE,
    INITIAL_CASH,
    LEAVE_IDLE,
    LOAN_AMOUNT,
    LOCATION_AUCTION_ISLAND,
    LOCATION_HARBOUR_OFFSET,
    LOCATION_OPEN_SEA,
    MAX_WAREHOUSES_PER_PLAYER,
    NO_OP,
    PRICE_SLOTS,
    PRODUCE_CHOICES,
    PURCHASE_STOP,
    SHIP_CAPACITY,
    ContainerFunctional,
    ContainerParams,
    EnvState,
    num_heads,
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers (used by test_container.py too — imported from there)
# ══════════════════════════════════════════════════════════════════════════════


def _make_func_env(num_players=2, num_colors=5):
    return ContainerFunctional(num_players=num_players, num_colors=num_colors)


def _make_params(num_players=2, num_colors=5):
    return ContainerParams(num_players=num_players, num_colors=num_colors)


def _make_state(**overrides):
    """Create an EnvState with defaults suitable for 2-player testing.

    P0 owns colour-0 factory with 1 container at slot 4 ($5).
    P1 owns colour-1 factory with 1 container at slot 4 ($5).
    All other stores / ships empty.  Both players have $20 cash, 1 warehouse.
    """
    np_, nc = 2, 5
    defaults = dict(
        cash=jnp.full((np_,), INITIAL_CASH, dtype=jnp.int32),
        loans=jnp.zeros((np_,), dtype=jnp.int32),
        factory_colors=jnp.zeros((np_, nc), dtype=jnp.int32).at[0, 0].set(1).at[1, 1].set(1),
        warehouse_count=jnp.ones((np_,), dtype=jnp.int32),
        factory_store=jnp.zeros((np_, nc, PRICE_SLOTS), dtype=jnp.int32).at[0, 0, 4].set(1).at[1, 1, 4].set(1),
        harbour_store=jnp.zeros((np_, nc, PRICE_SLOTS), dtype=jnp.int32),
        island_store=jnp.zeros((np_, nc), dtype=jnp.int32),
        ship_contents=jnp.zeros((np_, SHIP_CAPACITY), dtype=jnp.int32),
        ship_location=jnp.zeros((np_,), dtype=jnp.int32),
        container_supply=jnp.full((nc,), np_ * 4, dtype=jnp.int32),
        turn_phase=jnp.array(0, dtype=jnp.int32),
        current_player=jnp.array(0, dtype=jnp.int32),
        game_over=jnp.array(0, dtype=jnp.int32),
        secret_value_color=jnp.array([0, 1], dtype=jnp.int32),
        auction_active=jnp.array(0, dtype=jnp.int32),
        auction_seller=jnp.array(0, dtype=jnp.int32),
        auction_cargo=jnp.zeros(SHIP_CAPACITY, dtype=jnp.int32),
        auction_bids=jnp.zeros((np_,), dtype=jnp.int32),
        auction_round=jnp.array(0, dtype=jnp.int32),
        actions_taken=jnp.array(0, dtype=jnp.int32),
        produced_this_turn=jnp.array(0, dtype=jnp.int32),
        shopping_active=jnp.array(0, dtype=jnp.int32),
        shopping_action_type=jnp.array(0, dtype=jnp.int32),
        shopping_target=jnp.array(0, dtype=jnp.int32),
        shopping_harbour_price=jnp.array(0, dtype=jnp.int32),
        produce_active=jnp.array(0, dtype=jnp.int32),
        produce_pending=jnp.zeros((nc,), dtype=jnp.int32),
        produce_was_produced=jnp.array(0, dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    defaults.update(overrides)
    return EnvState(**defaults)


def _build_multihd(
    action_type: int, params: dict | None = None,
    num_players: int = 2, num_colors: int = 5,
) -> jnp.ndarray:
    """Build a v3 multi-head action array from *action_type* and optional *params*.

    All sub-head indices are offset +1 for the no-op element at index 0.
    Purchase defaults to ``PURCHASE_STOP`` (31).

    Shopping actions (3, 4) produce the **step-1** opponent-selection
    action only; continuation purchases must be built inline by the caller.
    """
    params = params or {}
    nc = num_colors
    np_ = num_players
    num_hds = num_heads(np_)
    mh = jnp.full(num_hds, PURCHASE_STOP, dtype=jnp.int32)
    mh = mh.at[HEAD_ACTION_TYPE].set(action_type + 1)

    if action_type == ACTION_BUY_FACTORY:
        mh = mh.at[HEAD_COLOR].set(jnp.clip(params.get("color", 0), 0, nc - 1) + 1)

    elif action_type == ACTION_PRODUCE:
        mh = mh.at[HEAD_COLOR].set(jnp.clip(params.get("color", 0), 0, nc - 1) + 1)
        mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(params.get("price_slot", 0), 0, PRODUCE_CHOICES - 1) + 1)

    elif action_type in (ACTION_BUY_FROM_FACTORY_STORE, ACTION_MOVE_LOAD):
        opp_idx = params.get("opponent", 1) - 1
        mh = mh.at[HEAD_OPPONENT].set(jnp.clip(opp_idx, 0, np_ - 2) + 1)

    elif action_type == ACTION_DOMESTIC_SALE:
        mh = mh.at[HEAD_COLOR].set(jnp.clip(params.get("color", 0), 0, nc - 1) + 1)
        mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(params.get("price_slot", 0), 0, PRICE_SLOTS - 1) + 1)

    return mh


def _rel_to_multihd(action_type: int, rel_offset: int, num_players: int = 2, num_colors: int = 5) -> jnp.ndarray:
    """Convert an old-style (action_type, rel_offset) to a v3 multi-head array.

    Shopping actions (3, 4) produce the step‑1 opponent‑selection action only.
    """
    nc = num_colors
    np_ = num_players
    combos = nc * PRICE_SLOTS
    num_hds = num_heads(np_)
    mh = jnp.full(num_hds, PURCHASE_STOP, dtype=jnp.int32)
    mh = mh.at[HEAD_ACTION_TYPE].set(action_type + 1)

    if action_type == ACTION_BUY_FACTORY:
        mh = mh.at[HEAD_COLOR].set(jnp.clip(rel_offset, 0, nc - 1) + 1)

    elif action_type in (ACTION_BUY_FROM_FACTORY_STORE, ACTION_MOVE_LOAD):
        opp_idx = rel_offset // combos
        mh = mh.at[HEAD_OPPONENT].set(jnp.clip(opp_idx, 0, np_ - 2) + 1)

    elif action_type == ACTION_DOMESTIC_SALE:
        remainder = rel_offset % combos
        color = remainder // PRICE_SLOTS
        price_slot = remainder % PRICE_SLOTS
        mh = mh.at[HEAD_COLOR].set(jnp.clip(color, 0, nc - 1) + 1)
        mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(price_slot, 0, PRICE_SLOTS - 1) + 1)

    return mh


# ══════════════════════════════════════════════════════════════════════════════
# Action 0 — Buy Factory
# ══════════════════════════════════════════════════════════════════════════════


class TestBuyFactory:
    """Buying a factory costs ``(factories_owned + 1) * 3`` dollars.

    The agent cannot buy a colour they already own nor exceed 5 factories.
    The handler reads only ``HEAD_COLOR - 1``; all other heads are ignored
    (forced to no‑op by the mask during parallel mode, and their log‑prob
    contribution is zeroed in training).
    """

    def test_buy_new_colour(self):
        """Buy a factory of colour 2 (not yet owned).

        **Why**: verify the basic purchase path — cash decreases,
        the new factory colour is recorded.
        """
        func_env = _make_func_env()
        state = _make_state()
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 2, 2, 5)  # rel_offset=2 → colour 2
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.factory_colors[0, 2]) == 1
        # 2nd factory cost: (1 + 1) * 3 = 6
        assert int(new_state.cash[0]) == INITIAL_CASH - 6

    def test_cannot_buy_duplicate_colour(self):
        """Attempt to buy colour 0 — already owned.

        **Why**: the handler must be idempotent — no state change when
        the action is invalid.
        """
        func_env = _make_func_env()
        state = _make_state()
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 0, 2, 5)
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.cash[0]) == INITIAL_CASH
        assert int(jnp.sum(new_state.factory_colors[0])) == 1

    def test_cannot_buy_when_maxed_out(self):
        """All 5 colours already owned — purchase should be rejected silently."""
        func_env = _make_func_env()
        colors = jnp.ones((2, 5), dtype=jnp.int32)
        state = _make_state(factory_colors=colors)
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 2, 2, 5)
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.cash[0]) == INITIAL_CASH

    def test_cannot_buy_when_cant_afford(self):
        """Only $1 cash — cannot afford a $3+ factory."""
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([1, 20], dtype=jnp.int32))
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 2, 2, 5)
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.cash[0]) == 1
        assert int(new_state.factory_colors[0, 2]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Action 1 — Buy Warehouse
# ══════════════════════════════════════════════════════════════════════════════


class TestBuyWarehouse:
    """Buying a warehouse costs ``warehouses_owned + 3`` dollars.

    Max 5 warehouses.  The handler reads no sub‑heads — all other values
    are forced to no‑op during parallel mode.
    """

    def test_buy_warehouse(self):
        """Buy a second warehouse.

        **Why**: baseline purchase — cost deducted, warehouse count incremented.
        """
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_BUY_WAREHOUSE)
        new_state = func_env._action_buy_warehouse(state, mh)
        assert int(new_state.warehouse_count[0]) == 2
        assert int(new_state.cash[0]) == INITIAL_CASH - 4  # 1 + 3 = 4

    def test_cannot_buy_past_max(self):
        """Already at 5 warehouses — no change."""
        func_env = _make_func_env()
        state = _make_state(warehouse_count=jnp.array([MAX_WAREHOUSES_PER_PLAYER, 1], dtype=jnp.int32))
        mh = _build_multihd(ACTION_BUY_WAREHOUSE)
        new_state = func_env._action_buy_warehouse(state, mh)
        assert int(new_state.warehouse_count[0]) == MAX_WAREHOUSES_PER_PLAYER
        assert int(new_state.cash[0]) == INITIAL_CASH

    def test_cannot_buy_when_cant_afford(self):
        """$0 cash — warehouse stays at 1."""
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([0, 20], dtype=jnp.int32))
        mh = _build_multihd(ACTION_BUY_WAREHOUSE)
        new_state = func_env._action_buy_warehouse(state, mh)
        assert int(new_state.warehouse_count[0]) == 1
        assert int(new_state.cash[0]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Action 2 — Produce
# ══════════════════════════════════════════════════════════════════════════════


class TestProduce:
    """Produce is a recurrent action with two kinds of step:

    1. **Enter** (parallel) — pays $1 to the player on the right (union
       boss), sets ``produce_active = 1`` and snapshots owned factories
       into ``produce_pending``.  No container is produced yet.
    2. **Recurrent** (sequential) — one step per pending factory.  Both
       ``colour`` and ``price_slot`` heads are active.  The selected
       factory may be set to a price $1–$4 or left idle.
    """

    def _do_produce(self, func_env, state, params, color: int, price_slot: int | None = None):
        """Helper: enter produce mode, optionally process one factory."""
        enter_action = jnp.array(
            [ACTION_PRODUCE + 1, 0, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        state = func_env._action_produce(state, enter_action, params)
        if color is None:
            return state
        mh = jnp.array(
            [0, 0, color + 1, price_slot + 1, PURCHASE_STOP], dtype=jnp.int32,
        )
        return func_env._produce_shopping_step(state, mh, params)

    def test_produce_adds_container(self):
        """Enter produce mode, then process one factory at $2.

        **Why**: verifies the full enter‑then‑process flow for a single
        factory.  Cash decreases by $1 (union dues), factory store gains
        one container, supply decreases by one.
        """
        func_env = _make_func_env()
        state = _make_state()
        params = _make_params()
        p0_color = int(jnp.argmax(state.factory_colors[0]))
        initial_supply = int(state.container_supply[p0_color])

        new_state = self._do_produce(func_env, state, params, p0_color, 0)

        assert int(jnp.sum(new_state.factory_store[0])) == 2  # 1 initial + 1 new
        assert int(new_state.container_supply[p0_color]) == initial_supply - 1
        assert int(new_state.cash[0]) == INITIAL_CASH - 1
        assert int(new_state.cash[1]) == INITIAL_CASH + 1
        assert int(new_state.produced_this_turn) == 1

    def test_can_produce_multiple_colors_per_turn(self):
        """Process two factories in the same produce batch.

        **Why**: the $1 union dues are paid only once (on enter).
        Subsequent factories add containers without additional payment.
        The batch finishes when all pending factories are processed.
        """
        func_env = _make_func_env()
        state = _make_state()
        params = _make_params()
        state = state._replace(factory_colors=state.factory_colors.at[0, 2].set(1))

        # Enter produce mode (pays $1)
        enter_action = jnp.array(
            [ACTION_PRODUCE + 1, 0, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        state = func_env._action_produce(state, enter_action, params)
        before_cash = int(state.cash[0])

        # Factory 1: colour 0, price 1 ($2)
        state = func_env._produce_shopping_step(state, jnp.array(
            [0, 0, 1, 2, PURCHASE_STOP], dtype=jnp.int32), params)
        assert int(state.produce_active) == 1

        # Factory 2: colour 2, price 2 ($3) — no extra payment
        state = func_env._produce_shopping_step(state, jnp.array(
            [0, 0, 3, 3, PURCHASE_STOP], dtype=jnp.int32), params)
        assert int(state.produce_active) == 0  # batch done
        assert int(state.cash[0]) == before_cash

    def test_pays_right_player(self):
        """Player 1 takes the produce action — pays player 0 (right).

        **Why**: the union boss is always the player on the acting
        player's *right*, regardless of view.
        """
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state(current_player=jnp.array(1, dtype=jnp.int32))
        p1_color = int(jnp.argmax(state.factory_colors[1]))
        new_state = self._do_produce(func_env, state, params, p1_color, 0)
        assert int(new_state.cash[1]) == INITIAL_CASH - 1
        assert int(new_state.cash[0]) == INITIAL_CASH + 1

    def test_no_produce_when_storage_full(self):
        """Factory store at capacity (2 containers for 1 factory).

        **Why**: ``_action_produce`` detects no space → ``do_pay`` is
        False → no $1 charged.  ``_produce_shopping_step`` also rejects
        the produce (can_produce = False) → container NOT added.
        """
        func_env = _make_func_env()
        params = _make_params()
        full_store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        full_store = full_store.at[0, 0, 0].set(2)  # capacity = 2 for 1 factory
        state = _make_state(factory_store=full_store)
        new_state = self._do_produce(func_env, state, params, 0, 0)
        assert int(jnp.sum(new_state.factory_store[0])) == 2
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action 3 — Buy from Factory Store (recurrent shopping)
# ══════════════════════════════════════════════════════════════════════════════


class TestBuyFromFactoryStore:
    """Buy containers from another player's factory store.

    Step 1 (parallel): select opponent — only ``action_type`` and
    ``opponent`` heads active.  Enters ``shopping_active = 1``.

    Step 2+ (sequential): ``colour`` head (which colour to buy) and
    ``purchase`` head (harbour price $2–$6 via indices 1–5; STOP at 31)
    are active.  The environment auto-selects the **cheapest** source
    slot for the chosen colour.  Continues until STOP or no more
    affordable stock / space.
    """

    def test_buy_one_container(self):
        """Buy colour 1 from P1 at harbour price $3.

        **Why**: verifies the full two‑step flow — opponent selection
        followed by a colour + harbour price purchase.  The auto‑cheapest
        logic picks source slot 4 ($5) because that is the only slot P1
        has for colour 1 in the default state.
        """
        func_env = _make_func_env()
        state = _make_state()
        params = _make_params()

        # Step 1: select opponent (P1 = opp_idx 0, HEAD_OPPONENT = 1)
        opp_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_buy_from_factory_store(state, opp_action, params)
        assert int(new_state.shopping_active) == 1

        # Step 2: colour 1 (HEAD_COLOR = 2), harbour $3 (HEAD_PURCHASE = 2)
        buy_action = jnp.array(
            [0, 0, 2, 0, 2], dtype=jnp.int32,
        )
        new_state = func_env._shopping_step(new_state, buy_action, params)

        assert int(new_state.cash[0]) == INITIAL_CASH - 5  # paid $5 source
        assert int(new_state.cash[1]) == INITIAL_CASH + 5
        assert int(new_state.factory_store[1, 1, 4]) == 0  # source consumed
        # harbour slot 2 (0‑based) = $3
        assert int(new_state.harbour_store[0, 1, 2]) == 1

    def test_no_stock_no_purchase(self):
        """Opponent has stock — shopping should be active after opponent selection."""
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        opp_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_buy_from_factory_store(state, opp_action, params)
        assert int(new_state.shopping_active) == 1

    def test_cannot_buy_from_self(self):
        """Design guarantee — ``_get_target_player`` never returns the acting player."""
        pass

    def test_cannot_buy_when_harbour_full(self):
        """Harbour at capacity (1 warehouse → 1 container).

        **Why**: the ``_can_continue_shopping`` check runs before entering
        shopping mode.  If no space, ``shopping_active`` stays 0 and no
        purchase is possible.
        """
        func_env = _make_func_env()
        params = _make_params()
        full_harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        full_harbour = full_harbour.at[0, 3, 0].set(1)
        state = _make_state(harbour_store=full_harbour)

        opp_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_buy_from_factory_store(state, opp_action, params)
        assert int(new_state.shopping_active) == 0
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action 4 — Move to Harbour + Load (recurrent)
# ══════════════════════════════════════════════════════════════════════════════


class TestMoveLoad:
    """Move ship to another player's harbour and load containers.

    Step 1 (parallel): select opponent — enters ``shopping_active``.

    Step 2+ (sequential): ``colour`` head (which colour) + ``purchase``
    head (1 = buy, 31 = STOP).  Environment auto‑cheapest source slot.
    Containers go directly to the ship (no harbour price needed).
    Ship location is set on the first load.
    """

    def test_load_from_harbour(self):
        """Load colour 2 from P1's harbour onto an empty ship.

        **Why**: verifies the two‑step flow, correct cash transfer,
        source depletion, ship placement, and location update.
        """
        func_env = _make_func_env()
        params = _make_params()
        harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        harbour = harbour.at[1, 2, 3].set(1)
        state = _make_state(harbour_store=harbour)

        # Step 1: opponent
        opp_action = jnp.array(
            [ACTION_MOVE_LOAD + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_move_load(state, opp_action, params)
        assert int(new_state.shopping_active) == 1

        # Step 2: colour 2 (HEAD_COLOR=3), buy signal (HEAD_PURCHASE=1)
        buy_action = jnp.array(
            [0, 0, 3, 0, 1], dtype=jnp.int32,
        )
        new_state = func_env._shopping_step(new_state, buy_action, params)

        assert int(new_state.cash[0]) == INITIAL_CASH - 4  # cost $4 (slot 3+1)
        assert int(new_state.cash[1]) == INITIAL_CASH + 4
        assert int(new_state.harbour_store[1, 2, 3]) == 0
        assert int(jnp.sum(new_state.ship_contents[0] > 0)) == 1
        # colour 2 → ship value = 3 (colour index + 1)
        assert int(new_state.ship_contents[0, 0]) == 3
        assert int(new_state.ship_location[0]) == LOCATION_HARBOUR_OFFSET + 1

    def test_cannot_load_from_empty_harbour(self):
        """Opponent has no harbour stock — shopping not entered."""
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        opp_action = jnp.array(
            [ACTION_MOVE_LOAD + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_move_load(state, opp_action, params)
        assert int(new_state.shopping_active) == 0

    def test_cannot_load_when_ship_full(self):
        """Ship is full (5/5) — ``_can_continue_shopping`` returns False."""
        func_env = _make_func_env()
        params = _make_params()
        full_ship = jnp.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        harbour = harbour.at[1, 0, 0].set(1)
        state = _make_state(ship_contents=full_ship, harbour_store=harbour)
        opp_action = jnp.array(
            [ACTION_MOVE_LOAD + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_move_load(state, opp_action, params)
        assert int(new_state.shopping_active) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Action 5 — Move to Open Sea
# ══════════════════════════════════════════════════════════════════════════════


class TestMoveSea:
    """Move the ship from any location to the open sea.

    Single‑step parallel action.  No sub‑heads used.
    """

    def test_move_to_sea(self):
        """Ship is in P1's harbour → moves to open sea.

        **Why**: verifies location changes from harbour back to sea.
        """
        func_env = _make_func_env()
        state = _make_state(
            ship_location=jnp.array([LOCATION_HARBOUR_OFFSET + 1, LOCATION_OPEN_SEA], dtype=jnp.int32),
        )
        mh = _build_multihd(ACTION_MOVE_SEA)
        new_state = func_env._action_move_sea(state, mh)
        assert int(new_state.ship_location[0]) == LOCATION_OPEN_SEA

    def test_move_to_sea_when_already_there(self):
        """Already at open sea — action is harmless (idempotent)."""
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_MOVE_SEA)
        new_state = func_env._action_move_sea(state, mh)
        assert int(new_state.ship_location[0]) == LOCATION_OPEN_SEA


# ══════════════════════════════════════════════════════════════════════════════
# Action 6 — Move to Auction Island + Hold Auction
# ══════════════════════════════════════════════════════════════════════════════


class TestAuction:
    """Initiate an auction.

    The ship must be at sea and carry cargo.  On success the cargo is
    snapshotted into ``auction_cargo``, the ship is cleared, and
    ``auction_active`` is set.  The current player advances to the first
    bidder.  Bidding and seller decision are tested separately in the
    recurrent auction flow.
    """

    def test_cannot_auction_when_not_at_sea(self):
        """Ship is in harbour — cargo stays on ship."""
        func_env = _make_func_env()
        params = _make_params()
        ship = jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(
            ship_contents=ship,
            ship_location=jnp.array([LOCATION_HARBOUR_OFFSET + 1, LOCATION_OPEN_SEA], dtype=jnp.int32),
        )
        key = random.PRNGKey(99)
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        new_state = func_env._action_move_auction(state, mh, key, params)
        assert int(new_state.ship_contents[0, 0]) == 1  # cargo not cleared

    def test_cannot_auction_empty_ship(self):
        """No cargo — auction not initiated, ship stays put."""
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        key = random.PRNGKey(99)
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        new_state = func_env._action_move_auction(state, mh, key, params)
        assert int(new_state.ship_location[0]) == LOCATION_OPEN_SEA

    def test_auction_clears_ship(self):
        """On success, the ship is emptied and moved to auction island."""
        func_env = _make_func_env()
        params = _make_params()
        ship = jnp.array([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(ship_contents=ship)
        key = random.PRNGKey(99)
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        new_state = func_env._action_move_auction(state, mh, key, params)
        assert int(jnp.sum(new_state.ship_contents[0] > 0)) == 0
        assert int(new_state.ship_location[0]) == LOCATION_AUCTION_ISLAND  # (defined in container.py)

    def test_auction_deposits_goods(self):
        """Full auction flow: initiate → bid → seller accepts → goods land on island.

        **Why**: the auction is a recurrent action spread across multiple
        steps.  Calling only ``_action_move_auction`` initiates the
        auction but does not resolve it.  We must manually step through
        bidding and seller decision via ``_auction_continue_step`` to
        verify the full deposit.
        """
        func_env = _make_func_env()
        params = _make_params()
        ship = jnp.array([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(ship_contents=ship)
        key = random.PRNGKey(99)

        # ── 1. Initiate auction (P0, with two containers) ───────────────
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        state = func_env._action_move_auction(state, mh, key, params)
        assert int(state.auction_active) == 1
        assert int(state.auction_seller) == 0

        # ── 2. P1 bids $5 ──────────────────────────────────────────────
        # HEAD_OPPONENT = 1 identifies P1 as the bidder
        key, subkey = random.split(key)
        bid_action = jnp.array(
            [ACTION_MOVE_AUCTION + 1, 1, 0, 0, 5], dtype=jnp.int32,
        )
        state = func_env._auction_continue_step(state, bid_action, subkey, params)
        # After P1's bid, auction_round → 1
        assert int(state.auction_round) == 1

        # ── 3. P0 (seller) accepts ─────────────────────────────────────
        # HEAD_OPPONENT = 0 identifies P0 as the seller acting
        key, subkey = random.split(key)
        accept_action = jnp.array(
            [ACTION_MOVE_AUCTION + 1, 0, 0, 0, 1], dtype=jnp.int32,  # 1 = accept
        )
        state = func_env._auction_continue_step(state, accept_action, subkey, params)
        assert int(state.auction_active) == 0  # auction resolved

        # ── 4. Goods deposited on island ────────────────────────────────
        total_island = int(jnp.sum(state.island_store))
        assert total_island == 2


# ══════════════════════════════════════════════════════════════════════════════
# Action 7 — Pass
# ══════════════════════════════════════════════════════════════════════════════


class TestPass:
    """Pass — does nothing.  No sub‑heads used."""

    def test_pass_no_change(self):
        """State is unchanged after a pass action."""
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_PASS)
        new_state = func_env._action_pass(state, mh)
        assert int(new_state.cash[0]) == int(state.cash[0])


# ══════════════════════════════════════════════════════════════════════════════
# Action 8 — Take Loan
# ══════════════════════════════════════════════════════════════════════════════


class TestTakeLoan:
    """Take a $10 loan.  Max 2 outstanding loans.  Does NOT consume an action."""

    def test_take_loan(self):
        """Baseline: cash +$10, loans +1."""
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_TAKE_LOAN)
        new_state = func_env._action_take_loan(state, mh)
        assert int(new_state.loans[0]) == 1
        assert int(new_state.cash[0]) == INITIAL_CASH + LOAN_AMOUNT

    def test_cannot_exceed_two_loans(self):
        """Already at 2 loans — state unchanged."""
        func_env = _make_func_env()
        state = _make_state(loans=jnp.array([2, 0], dtype=jnp.int32))
        mh = _build_multihd(ACTION_TAKE_LOAN)
        new_state = func_env._action_take_loan(state, mh)
        assert int(new_state.loans[0]) == 2
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action 9 — Repay Loan
# ══════════════════════════════════════════════════════════════════════════════


class TestRepayLoan:
    """Repay a $10 loan.  Requires at least 1 loan and enough cash."""

    def test_repay_loan(self):
        """Repay 1 of 1 loan — loans → 0, cash −$10."""
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([40, 20], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
        )
        mh = _build_multihd(ACTION_REPAY_LOAN)
        new_state = func_env._action_repay_loan(state, mh)
        assert int(new_state.loans[0]) == 0
        assert int(new_state.cash[0]) == 40 - LOAN_AMOUNT

    def test_cannot_repay_without_loan(self):
        """No loan — nothing changes."""
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_REPAY_LOAN)
        new_state = func_env._action_repay_loan(state, mh)
        assert int(new_state.loans[0]) == 0

    def test_cannot_repay_without_cash(self):
        """$5 cash, $10 loan — cannot afford, nothing changes."""
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([5, 20], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
        )
        mh = _build_multihd(ACTION_REPAY_LOAN)
        new_state = func_env._action_repay_loan(state, mh)
        assert int(new_state.loans[0]) == 1
        assert int(new_state.cash[0]) == 5


# ══════════════════════════════════════════════════════════════════════════════
# Action 10 — Domestic Sale (variant)
# ══════════════════════════════════════════════════════════════════════════════


class TestDomesticSale:
    """Sell one container back to the supply for $2.

    Prefers factory store first, then falls back to harbour store.
    Uses ``colour`` and ``price_slot`` heads; all other heads forced no‑op.
    """

    def test_sell_from_factory_store(self):
        """Sell colour 0 from factory slot 4.

        **Why**: baseline — cash +$2, container removed, supply replenished.
        """
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        mh = _rel_to_multihd(ACTION_DOMESTIC_SALE, 0 * 50 + 0 * 10 + 4, 2, 5)
        new_state = func_env._action_domestic_sale(state, mh, params)
        assert int(new_state.cash[0]) == INITIAL_CASH + 2
        assert int(new_state.factory_store[0, 0, 4]) == 0
        assert int(new_state.container_supply[0]) == 9

    def test_sell_from_harbour_store(self):
        """Sell colour 2 from harbour slot 3 (factory store empty).

        **Why**: verifies the fallback to harbour when factory has nothing
        at the requested (colour, slot).
        """
        func_env = _make_func_env()
        params = _make_params()
        harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        harbour = harbour.at[0, 2, 3].set(1)
        state = _make_state(factory_store=jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32), harbour_store=harbour)
        mh = _rel_to_multihd(ACTION_DOMESTIC_SALE, 1 * 50 + 2 * 10 + 3, 2, 5)
        new_state = func_env._action_domestic_sale(state, mh, params)
        assert int(new_state.cash[0]) == INITIAL_CASH + 2
        assert int(new_state.harbour_store[0, 2, 3]) == 0
        assert int(new_state.container_supply[2]) == 8

    def test_factory_sale_falls_back_to_harbour(self):
        """Factory empty at (0, 4) — same (colour, slot) is taken from harbour.

        **Why**: domestic sale tries factory first; if that specific slot
        has 0 containers it checks the harbour.
        """
        func_env = _make_func_env()
        params = _make_params()
        harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        harbour = harbour.at[0, 0, 4].set(2)
        state = _make_state(
            factory_store=jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32),
            harbour_store=harbour,
        )
        mh = _rel_to_multihd(ACTION_DOMESTIC_SALE, 0 * 50 + 0 * 10 + 4, 2, 5)
        new_state = func_env._action_domestic_sale(state, mh, params)
        assert int(new_state.cash[0]) == INITIAL_CASH + 2
        assert int(new_state.harbour_store[0, 0, 4]) == 1

    def test_no_sale_when_empty(self):
        """Both factory and harbour empty — no cash change."""
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state(
            factory_store=jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32),
            harbour_store=jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32),
        )
        mh = _rel_to_multihd(ACTION_DOMESTIC_SALE, 0 * 50 + 0 * 10 + 0, 2, 5)
        new_state = func_env._action_domestic_sale(state, mh, params)
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action mask verification — per-mode, per-head correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestActionMasksParallel:
    """Verify masks in **parallel mode** (no shopping, no produce, no auction).

    In parallel mode every head's no‑op (index 0) is forced to 0.
    All legally selectable values at indices ≥ 1 have mask = 1,
    all illegal values have mask = 0.

    The default test state:
    - P0 owns colour‑0 factory, has 1 container of colour 0 at slot 4 ($5)
    - P1 owns colour‑1 factory, has 1 container of colour 1 at slot 4 ($5)
    - Both have $20 cash, 1 warehouse, empty harbour / island / ship
    """

    @staticmethod
    def _masks(state, params=None):
        func_env = _make_func_env()
        if params is None:
            params = _make_params()
        return func_env._action_masks(state, params)

    # ── Structural invariants ─────────────────────────────────────────

    def test_no_op_masked_out_on_all_heads(self):
        """Index 0 must be 0 on every head in parallel mode.

        **Why**: the agent must NOT be allowed to select no‑op during a
        normal turn — only meaningful values should be available.  No‑op
        is reserved for heads that are irrelevant in sequential
        continuation modes.
        """
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][0]) == 0, "action_type no‑op should be masked"
        assert int(masks["opponent"][0]) == 0, "opponent no‑op should be masked"
        assert int(masks["color"][0]) == 0, "colour no‑op should be masked"
        assert int(masks["price_slot"][0]) == 0, "price_slot no‑op should be masked"
        assert int(masks["purchase"][0]) == 0, "purchase no‑op should be masked"

    def test_head_sizes(self):
        """Mask arrays must match the declared MultiDiscrete head sizes.

        **Why**: the observation appends these masks and the training
        wrapper splits them by size — a mismatch here would cause silent
        misalignment during PPO training.
        """
        from container_rl.env.container import head_sizes
        state = _make_state()
        masks = self._masks(state)
        sizes = head_sizes(2, 5)  # 2 players, 5 colours → [12, 2, 6, 11, 32]
        assert len(masks["action_type"]) == sizes[0]
        assert len(masks["opponent"]) == sizes[1]
        assert len(masks["color"]) == sizes[2]
        assert len(masks["price_slot"]) == sizes[3]
        assert len(masks["purchase"]) == sizes[4]

    # ── Action-type legality ───────────────────────────────────────────

    def test_buy_factory_legal(self):
        """P0 can afford a new factory (cost $3), does NOT own all colours.

        **Why**: factory cost = (1 + 1) * 3 = $6.  P0 has $20 and only
        owns colour 0, so buy_factory (index 1) = 1.
        """
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_FACTORY + 1]) == 1

    def test_buy_factory_illegal_when_own_all(self):
        """P0 already owns all 5 colours — buy_factory masked out.

        **Why**: max factories = 5.  When own_all is True,
        ``action_type[1]`` must be 0.
        """
        state = _make_state(factory_colors=jnp.ones((2, 5), dtype=jnp.int32))
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_FACTORY + 1]) == 0

    def test_buy_factory_illegal_when_cant_afford(self):
        """P0 has $1 — cannot afford the cheapest factory ($3).

        **Why**: ``cash >= factory_cost`` is False, so the mask must be 0.
        """
        state = _make_state(cash=jnp.array([1, 20], dtype=jnp.int32))
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_FACTORY + 1]) == 0

    def test_buy_warehouse_legal(self):
        """P0 has 1 warehouse and $20 — can buy a 2nd ($4).

        **Why**: not at max, cash >= 4 → mask = 1.
        """
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_WAREHOUSE + 1]) == 1

    def test_buy_warehouse_illegal_at_max(self):
        """Already at 5 warehouses — mask must be 0."""
        state = _make_state(
            warehouse_count=jnp.array([MAX_WAREHOUSES_PER_PLAYER, 1], dtype=jnp.int32),
        )
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_WAREHOUSE + 1]) == 0

    def test_produce_legal(self):
        """P0 has a factory, space, and supply — produce must be legal.

        **Why**: not produced yet this turn, factory store has 1 of 2
        capacity, colour-0 supply > 0.
        """
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_PRODUCE + 1]) == 1

    def test_produce_illegal_when_already_produced(self):
        """Already produced this turn — mask must be 0."""
        state = _make_state(produced_this_turn=jnp.array(1, dtype=jnp.int32))
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_PRODUCE + 1]) == 0

    def test_produce_illegal_when_storage_full(self):
        """Factory store at capacity — mask must be 0."""
        full = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32).at[0, 0, 0].set(2)
        state = _make_state(factory_store=full)
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_PRODUCE + 1]) == 0

    def test_buy_from_factory_store_legal(self):
        """P1 has affordable stock and P0 has harbour space.

        **Why**: P1's factory store has colour 1 at slot 4 ($5), P0
        has $20 and an empty harbour (capacity 1).
        """
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_FROM_FACTORY_STORE + 1]) == 1

    def test_buy_from_factory_store_illegal_harbour_full(self):
        """P0's harbour is full — cannot buy more."""
        full_harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32).at[0, 3, 0].set(1)
        state = _make_state(harbour_store=full_harbour)
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_BUY_FROM_FACTORY_STORE + 1]) == 0

    def test_move_load_legal(self):
        """P1 has affordable harbour stock and P0 has ship space.

        **Why**: need to give P1 harbour stock for this test — default
        state has none, so mask = 0 by default.  We create harbour stock.
        """
        harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32).at[1, 2, 3].set(1)
        state = _make_state(harbour_store=harbour)
        masks = self._masks(state)
        # P1 has colour 2 at slot 3 ($4), P0 has $20 + empty ship (5 cap)
        assert int(masks["action_type"][ACTION_MOVE_LOAD + 1]) == 1

    def test_move_sea_legal_when_in_harbour(self):
        """Ship in harbour — move to sea must be legal."""
        state = _make_state(
            ship_location=jnp.array([LOCATION_HARBOUR_OFFSET + 1, LOCATION_OPEN_SEA], dtype=jnp.int32),
        )
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_MOVE_SEA + 1]) == 1

    def test_move_sea_illegal_when_at_sea(self):
        """Ship already at sea — move_sea is redundant but still legal (idempotent)."""
        state = _make_state()
        masks = self._masks(state)
        # Currently in_harbour is False, but the handler is idempotent.
        # The mask allows it since the condition is ``in_harbour.astype(jnp.int32)``.
        # With ship at sea, in_harbour = False → mask = 0.
        assert int(masks["action_type"][ACTION_MOVE_SEA + 1]) == 0

    def test_auction_legal(self):
        """Ship at sea with cargo — auction must be legal."""
        ship = jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(ship_contents=ship)
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_MOVE_AUCTION + 1]) == 1

    def test_auction_illegal_empty_ship(self):
        """No cargo — auction not allowed."""
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_MOVE_AUCTION + 1]) == 0

    def test_pass_always_legal(self):
        """Pass is unconditionally legal."""
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_PASS + 1]) == 1

    def test_take_loan_legal(self):
        """Loans < 2 — take_loan must be legal."""
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_TAKE_LOAN + 1]) == 1

    def test_take_loan_illegal_at_max(self):
        """Already 2 loans — take_loan masked out."""
        state = _make_state(loans=jnp.array([2, 0], dtype=jnp.int32))
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_TAKE_LOAN + 1]) == 0

    def test_repay_loan_legal(self):
        """Has a loan and enough cash ($40) — repay must be legal."""
        state = _make_state(cash=jnp.array([40, 20], dtype=jnp.int32),
                            loans=jnp.array([1, 0], dtype=jnp.int32))
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_REPAY_LOAN + 1]) == 1

    def test_repay_loan_illegal_no_loan(self):
        """No outstanding loans — repay masked out."""
        state = _make_state()
        masks = self._masks(state)
        assert int(masks["action_type"][ACTION_REPAY_LOAN + 1]) == 0

    # ── Colour mask in parallel mode ───────────────────────────────────

    def test_color_mask_shows_buyable_colours(self):
        """P0 owns colour 0 — colours 1-4 should be buyable.

        **Why**: ``not_owned`` is True for colours 1-4, cost $6 ≤ $20.
        Indices 2-5 (= colours 1-4) should be 1, index 1 (= colour 0) = 0.
        """
        state = _make_state()
        masks = self._masks(state)
        cm = masks["color"]
        assert int(cm[0]) == 0, "no-op masked out"
        assert int(cm[1]) == 0, "colour 0 already owned"
        assert int(cm[2]) == 1, "colour 1 buyable"
        assert int(cm[3]) == 1, "colour 2 buyable"
        assert int(cm[4]) == 1, "colour 3 buyable"
        assert int(cm[5]) == 1, "colour 4 buyable"

    def test_color_mask_all_owned(self):
        """All colours owned — no colour options (all = 0 except no‑op)."""
        state = _make_state(factory_colors=jnp.ones((2, 5), dtype=jnp.int32))
        masks = self._masks(state)
        cm = masks["color"]
        for i in range(1, 6):
            assert int(cm[i]) == 0, f"colour {i - 1} should be unselectable"

    # ── Opponent mask in parallel mode ────────────────────────────────

    def test_opponent_mask_parallel(self):
        """P1 has factory stock — opponent must be selectable.

        **Why**: the opponent mask shows opponents with affordable stock
        at index 1 (P1).  P1 has colour-1 at slot 4 ($5 ≤ $20).
        """
        state = _make_state()
        masks = self._masks(state)
        om = masks["opponent"]
        assert int(om[0]) == 0, "no-op masked out"
        assert int(om[1]) == 1, "P1 has affordable stock"

    # ── Purchase mask in parallel mode ─────────────────────────────────

    def test_purchase_mask_parallel_all_values(self):
        """In parallel mode all value slots 1-30 + STOP (31) are valid.

        **Why**: the purchase head is a generic value slot — the
        actual meaning depends on which action is chosen.  So all
        non‑no‑op slots are available.
        """
        state = _make_state()
        masks = self._masks(state)
        pm = masks["purchase"]
        assert int(pm[0]) == 0, "no-op masked out"
        for i in range(1, 31):
            assert int(pm[i]) == 1, f"purchase slot {i} should be valid"
        assert int(pm[31]) == 1, "STOP always valid"


# ══════════════════════════════════════════════════════════════════════════════


class TestActionMasksShopping:
    """Verify masks during **shopping continuation** (``shopping_active = 1``).

    Two sub‑modes exist:
    - **Factory store** (``shopping_action_type == 3``): colour + harbour
      price ($2-$6) heads active.
    - **Ship load** (``shopping_action_type == 4``): colour + buy‑signal
      head active.
    """

    @staticmethod
    def _masks(state, params=None):
        func_env = _make_func_env()
        if params is None:
            params = _make_params()
        return func_env._action_masks(state, params)

    def _shopping_state(self, action_type, target=1):
        """Create a state mid‑shopping with given target opponent."""
        harbour_store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        factory_store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        if action_type == ACTION_BUY_FROM_FACTORY_STORE:
            # P0 buys from P1 factory: put stock for P1
            factory_store = factory_store.at[1, 1, 4].set(1)  # colour 1, slot 4 ($5)
        else:
            # MOVE_LOAD: P1 harbour stock
            harbour_store = harbour_store.at[1, 2, 3].set(1)  # colour 2, slot 3 ($4)
        return _make_state(
            shopping_active=jnp.array(1, dtype=jnp.int32),
            shopping_action_type=jnp.array(action_type, dtype=jnp.int32),
            shopping_target=jnp.array(target, dtype=jnp.int32),
            factory_store=factory_store,
            harbour_store=harbour_store,
        )

    def test_forced_no_op_on_action_type(self):
        """Action type head MUST be forced to no‑op only during shopping.

        **Why**: the action type was already selected in step 1; the
        agent should not re‑select it.  Only index 0 = 1.
        """
        state = self._shopping_state(ACTION_BUY_FROM_FACTORY_STORE)
        masks = self._masks(state)
        at = masks["action_type"]
        assert int(at[0]) == 1, "no-op forced"
        for i in range(1, len(at)):
            assert int(at[i]) == 0, f"action_type[{i}] should be forced off"

    def test_forced_no_op_on_opponent(self):
        """Opponent head forced to no‑op during shopping.

        **Why**: the opponent was locked in step 1 via ``shopping_target``.
        """
        state = self._shopping_state(ACTION_BUY_FROM_FACTORY_STORE)
        masks = self._masks(state)
        om = masks["opponent"]
        assert int(om[0]) == 1, "no-op forced"
        for i in range(1, len(om)):
            assert int(om[i]) == 0

    def test_forced_no_op_on_price_slot(self):
        """Price-slot head forced to no‑op during shopping.

        **Why**: the cheapest source slot is auto‑selected by the
        environment; the harbour price comes from the **purchase** head.
        """
        state = self._shopping_state(ACTION_BUY_FROM_FACTORY_STORE)
        masks = self._masks(state)
        sm = masks["price_slot"]
        assert int(sm[0]) == 1
        for i in range(1, len(sm)):
            assert int(sm[i]) == 0

    def test_colour_active_factory_shop(self):
        """Colour head shows available colours from the target opponent.

        **Why**: P1 factory has colour 1 at slot 4 → only colour 1 should
        be selectable (index 2), others masked out.
        """
        state = self._shopping_state(ACTION_BUY_FROM_FACTORY_STORE)
        masks = self._masks(state)
        cm = masks["color"]
        assert int(cm[0]) == 0, "no-op masked out on active head"
        assert int(cm[2]) == 1, "colour 1 available from P1 factory"
        assert int(cm[1]) == 0, "colour 0 not available"
        assert int(cm[3]) == 0, "colour 2 not available"

    def test_colour_active_ship_shop(self):
        """Colour head shows available colours from target's harbour.

        **Why**: P1 harbour has colour 2 at slot 3 → index 3 (= colour 2) = 1.
        """
        state = self._shopping_state(ACTION_MOVE_LOAD)
        masks = self._masks(state)
        cm = masks["color"]
        assert int(cm[0]) == 0, "no-op masked out"
        assert int(cm[3]) == 1, "colour 2 available from P1 harbour"
        assert int(cm[2]) == 0, "colour 1 not available"

    def test_purchase_factory_shop_harbour_prices(self):
        """During factory shopping purchase head shows $2-$6 + STOP.

        **Why**: indices 1-5 = harbour $2-$6, index 31 = STOP.
        All other indices (6-30) must be 0.
        """
        state = self._shopping_state(ACTION_BUY_FROM_FACTORY_STORE)
        masks = self._masks(state)
        pm = masks["purchase"]
        assert int(pm[0]) == 0, "no-op masked out"
        for i in range(1, 6):
            assert int(pm[i]) == 1, f"harbour price index {i} should be valid"
        for i in range(6, 31):
            assert int(pm[i]) == 0, f"index {i} should be masked"
        assert int(pm[31]) == 1, "STOP valid"

    def test_purchase_ship_shop_buy_or_stop(self):
        """During ship shopping purchase head shows only 'buy' (1) + STOP (31).

        **Why**: containers go to ship — no harbour price needed.
        Index 1 = "buy this colour", index 31 = STOP.
        """
        state = self._shopping_state(ACTION_MOVE_LOAD)
        masks = self._masks(state)
        pm = masks["purchase"]
        assert int(pm[0]) == 0, "no-op masked out"
        assert int(pm[1]) == 1, "buy signal"
        for i in range(2, 31):
            assert int(pm[i]) == 0, f"index {i} should be masked (ship shop)"
        assert int(pm[31]) == 1, "STOP valid"


# ══════════════════════════════════════════════════════════════════════════════


class TestActionMasksProduce:
    """Verify masks during **produce continuation** (``produce_active = 1``).

    Only ``colour`` and ``price_slot`` heads are active.  All others are
    forced to no‑op only.
    """

    @staticmethod
    def _masks(state, params=None):
        func_env = _make_func_env()
        if params is None:
            params = _make_params()
        return func_env._action_masks(state, params)

    def _produce_state(self):
        """State mid‑produce with colour 0 and colour 2 pending.

        P0 owns colours 0 and 2; both are pending (= 1 in produce_pending).
        """
        return _make_state(
            produce_active=jnp.array(1, dtype=jnp.int32),
            produce_pending=jnp.array([1, 0, 1, 0, 0], dtype=jnp.int32),
            factory_colors=jnp.zeros((2, 5), dtype=jnp.int32).at[0, 0].set(1).at[0, 2].set(1).at[1, 1].set(1),
        )

    def test_forced_no_op_on_action_type_opponent_purchase(self):
        """Action type, opponent, and purchase heads forced to no‑op only.

        **Why**: during produce continuation the only decisions are
        *which factory* (colour) and *at what price* (price_slot).
        """
        state = self._produce_state()
        masks = self._masks(state)

        at = masks["action_type"]
        assert int(at[0]) == 1, "action_type forced no-op"
        assert all(int(at[i]) == 0 for i in range(1, len(at)))

        om = masks["opponent"]
        assert int(om[0]) == 1, "opponent forced no-op"
        assert all(int(om[i]) == 0 for i in range(1, len(om)))

        pm = masks["purchase"]
        assert int(pm[0]) == 1, "purchase forced no-op"
        assert all(int(pm[i]) == 0 for i in range(1, len(pm)))

    def test_colour_shows_pending_factories(self):
        """Only pending factories (colours 0, 2) are selectable.

        **Why**: ``produce_pending = [1, 0, 1, 0, 0]`` → indices 1 and 3
        (= colours 0, 2) should be 1.  Colours 1, 3, 4 should be 0.
        """
        state = self._produce_state()
        masks = self._masks(state)
        cm = masks["color"]
        assert int(cm[0]) == 0, "no-op masked out on active head"
        assert int(cm[1]) == 1, "colour 0 pending"
        assert int(cm[2]) == 0, "colour 1 not owned/not pending"
        assert int(cm[3]) == 1, "colour 2 pending"
        assert int(cm[4]) == 0, "colour 3 not owned"
        assert int(cm[5]) == 0, "colour 4 not owned"

    def test_price_slot_produce_range(self):
        """Only $1-$4 (indices 1-4) + leave idle (index 5) are valid.

        **Why**: ``PRODUCE_CHOICES = 5`` (4 prices + leave idle).
        Indices 1-5 must be 1, indices 6-10 must be 0.
        """
        state = self._produce_state()
        masks = self._masks(state)
        sm = masks["price_slot"]
        assert int(sm[0]) == 0, "no-op masked out"
        for i in range(1, 6):  # indices 1-5 = slots 0-4
            assert int(sm[i]) == 1, f"produce slot {i - 1} should be valid"
        for i in range(6, 11):
            assert int(sm[i]) == 0, f"slot {i - 1} should be masked"


# ══════════════════════════════════════════════════════════════════════════════


class TestActionMasksAuction:
    """Verify masks during **auction** (``auction_active = 1``).

    Two sub‑modes:
    - **Bidding** (current player ≠ seller): action_type locked to
      AUCTION, purchase shows bid amounts 0..cash.
    - **Seller decision** (current player == seller): action_type locked,
      purchase shows reject (0) or accept (1).
    """

    @staticmethod
    def _masks(state, params=None):
        func_env = _make_func_env()
        if params is None:
            params = _make_params()
        return func_env._action_masks(state, params)

    def _auction_state(self, current_player=1, seller=0, cash=20):
        """State mid‑auction.  Default: P1 is bidding, P0 is seller."""
        return _make_state(
            auction_active=jnp.array(1, dtype=jnp.int32),
            auction_seller=jnp.array(seller, dtype=jnp.int32),
            auction_cargo=jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
            auction_bids=jnp.array([0, -1, -1, -1], dtype=jnp.int32),  # P0 done, P1 pending
            current_player=jnp.array(current_player, dtype=jnp.int32),
            cash=jnp.array([cash, cash], dtype=jnp.int32),
        )

    def test_action_type_locked_to_auction(self):
        """Only AUCTION is selectable on the action_type head.

        **Why**: during auction, players cannot take any other action type.
        Index 7 (= ACTION_MOVE_AUCTION + 1) must be 1, all others —including
        no‑op (0)— must be 0.  The agent is forced to pick AUCTION.
        """
        state = self._auction_state(current_player=1, seller=0)
        masks = self._masks(state)
        at = masks["action_type"]
        assert int(at[ACTION_MOVE_AUCTION + 1]) == 1, "AUCTION should be valid"
        assert int(at[0]) == 0, "no‑op masked out (AUCTION forced)"

    def test_opponent_color_price_forced_no_op(self):
        """Colour and price_slot heads forced to no‑op during auction.

        Opponent head is repurposed as direct player index during auction
        (identifies which player is acting), so it is NOT forced no‑op.
        """
        state = self._auction_state()
        masks = self._masks(state)
        for key in ("color", "price_slot"):
            m = masks[key]
            assert int(m[0]) == 1, f"{key} no-op forced"
            assert all(int(m[i]) == 0 for i in range(1, len(m))), \
                f"{key} all non‑no‑op should be masked"
        # Opponent head: all player indices valid during auction
        om = masks["opponent"]
        assert int(om[0]) == 1, "opponent index 0 (P0) valid during auction"
        assert int(om[1]) == 1, "opponent index 1 (P1) valid during auction"

    def test_purchase_bidding_shows_cash_range(self):
        """During bidding, purchase head shows $0 bid + $1..cash bids.

        **Why**: index 0 = $0 bid, indices 1..cash = valid bid amounts.
        STOP (31) must NOT be shown.
        """
        state = self._auction_state(current_player=1, seller=0, cash=20)
        masks = self._masks(state)
        pm = masks["purchase"]
        assert int(pm[0]) == 1, "$0 bid valid"
        for i in range(1, 21):
            assert int(pm[i]) == 1, f"${i} bid should be valid"
        for i in range(21, 31):
            assert int(pm[i]) == 0, f"${i} exceeds cash → masked"
        assert int(pm[31]) == 0, "STOP not valid during auction"

    def test_purchase_bidding_respects_low_cash(self):
        """Player has only $5 — bid range must be $0..$5 only."""
        state = self._auction_state(current_player=1, seller=0, cash=5)
        masks = self._masks(state)
        pm = masks["purchase"]
        assert int(pm[0]) == 1, "$0 bid valid"
        for i in range(1, 6):
            assert int(pm[i]) == 1, f"${i} bid valid"
        for i in range(6, 32):
            assert int(pm[i]) == 0, f"index {i} should be masked"

    def test_purchase_seller_decision(self):
        """Seller (P0) sees only reject (index 0) or accept (index 1).

        **Why**: the seller cannot bid — they only decide to accept or
        reject the highest bid.
        """
        state = self._auction_state(current_player=0, seller=0, cash=20)
        masks = self._masks(state)
        pm = masks["purchase"]
        assert int(pm[0]) == 1, "reject ($0) valid"
        assert int(pm[1]) == 1, "accept valid"
        for i in range(2, 32):
            assert int(pm[i]) == 0, f"index {i} should be masked for seller"


# ══════════════════════════════════════════════════════════════════════════════


class TestActionMasksTransitions:
    """Verify that masks change correctly when modes transition."""

    @staticmethod
    def _masks(state, params=None):
        func_env = _make_func_env()
        if params is None:
            params = _make_params()
        return func_env._action_masks(state, params)

    def test_parallel_to_produce_transition(self):
        """Entering produce mode must switch from parallel to produce masks.

        **Why**: before enter‑produce, all heads active (parallel).
        After, only colour+price_slot active (produce).
        """
        state = _make_state()
        # Parallel
        p_masks = self._masks(state)
        assert int(p_masks["action_type"][0]) == 0, "parallel: no‑op masked on action_type"
        assert int(p_masks["action_type"][ACTION_PRODUCE + 1]) == 1, "parallel: produce legal"

        # After entering produce mode
        state = state._replace(
            produce_active=jnp.array(1, dtype=jnp.int32),
            produce_pending=jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
        )
        c_masks = self._masks(state)
        assert int(c_masks["action_type"][0]) == 1, "produce: action_type forced no‑op"
        assert int(c_masks["color"][0]) == 0, "produce: colour active (no‑op masked)"
        assert int(c_masks["price_slot"][0]) == 0, "produce: price_slot active"

    def test_parallel_to_shopping_transition(self):
        """Entering shopping mode must switch from parallel to shopping masks.

        **Why**: after opponent selection, only colour+purchase active.
        """
        state = _make_state()
        # Parallel
        p_masks = self._masks(state)
        assert int(p_masks["opponent"][0]) == 0, "parallel: opponent no‑op masked"

        # After entering shopping (factory)
        state = state._replace(
            shopping_active=jnp.array(1, dtype=jnp.int32),
            shopping_action_type=jnp.array(ACTION_BUY_FROM_FACTORY_STORE, dtype=jnp.int32),
            shopping_target=jnp.array(1, dtype=jnp.int32),
            factory_store=jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32).at[1, 1, 4].set(1),
        )
        s_masks = self._masks(state)
        assert int(s_masks["action_type"][0]) == 1, "shopping: action_type forced no‑op"
        assert int(s_masks["opponent"][0]) == 1, "shopping: opponent forced no‑op"
        assert int(s_masks["color"][0]) == 0, "shopping: colour active"
        assert int(s_masks["purchase"][0]) == 0, "shopping: purchase active"
        assert int(s_masks["price_slot"][0]) == 1, "shopping: price_slot forced no‑op"

    def test_parallel_to_auction_transition(self):
        """Entering auction must lock action_type to AUCTION and show bid mask.

        **Why**: during bidding only AUCTION is legal; purchase shows bids.
        """
        state = _make_state()
        p_masks = self._masks(state)
        assert int(p_masks["action_type"][ACTION_MOVE_AUCTION + 1]) == 0, \
            "parallel: auction illegal (no cargo)"

        # After initiating auction (P1 is first bidder)
        state = state._replace(
            auction_active=jnp.array(1, dtype=jnp.int32),
            auction_seller=jnp.array(0, dtype=jnp.int32),
            auction_cargo=jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
            auction_bids=jnp.array([0, -1, -1, -1], dtype=jnp.int32),
            current_player=jnp.array(1, dtype=jnp.int32),
        )
        a_masks = self._masks(state)
        assert int(a_masks["action_type"][ACTION_MOVE_AUCTION + 1]) == 1, \
            "auction: AUCTION forced"
        assert int(a_masks["action_type"][0]) == 0, \
            "auction: action_type no‑op masked (AUCTION forced)"
        assert int(a_masks["purchase"][0]) == 1, \
            "auction: $0 bid valid"
