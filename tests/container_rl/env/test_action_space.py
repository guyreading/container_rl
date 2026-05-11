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
        """After resolution the cargo lands on the island store.

        **Why**: integration check — cargo is transferred via
        ``_deposit_goods`` during ``_auction_continue_step`` resolution.
        Since JIT is disabled, the immediate step resolves and goods
        appear in island stores.
        """
        func_env = _make_func_env()
        params = _make_params()
        ship = jnp.array([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(ship_contents=ship)
        key = random.PRNGKey(99)
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        new_state = func_env._action_move_auction(state, mh, key, params)
        total_island = int(jnp.sum(new_state.island_store))
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
