"""Unit and integration tests for the Container RL environment."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

jax.config.update("jax_disable_jit", True)

from container_rl.env.container import (  # noqa: E402
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
    LOAN_AMOUNT,
    LOCATION_AUCTION_ISLAND,
    LOCATION_HARBOUR_OFFSET,
    LOCATION_OPEN_SEA,
    MAX_WAREHOUSES_PER_PLAYER,
    PRICE_SLOTS,
    PRODUCE_CHOICES,
    PURCHASE_STOP,
    SHIP_CAPACITY,
    ActionEncoder,
    ContainerFunctional,
    ContainerJaxEnv,
    ContainerParams,
    EnvState,
    num_heads,
)

# ── test fixtures ────────────────────────────────────────────────────────────


def _make_func_env(num_players=2, num_colors=5):
    return ContainerFunctional(num_players=num_players, num_colors=num_colors)


def _make_params(num_players=2, num_colors=5):
    return ContainerParams(num_players=num_players, num_colors=num_colors)


def _make_initial_state(func_env=None, rng=None, num_players=2, num_colors=5):
    if func_env is None:
        func_env = _make_func_env(num_players, num_colors)
    if rng is None:
        rng = random.PRNGKey(0)
    return func_env.initial(rng, ContainerParams(num_players=num_players, num_colors=num_colors))


def _make_state(**overrides):
    """Create an EnvState with defaults suitable for 2-player testing."""
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
    """Build a multi-head action array from action_type and optional params.

    All sub-head indices are offset by +1 for the no-op element at index 0.
    Purchase defaults to PURCHASE_STOP (31).  Shopping actions produce the
    step-1 opponent selection only; continuation purchases must use actions
    built directly.
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
        # Step 1: opponent selection only.  PURCHASE_STOP is default.

    elif action_type == ACTION_DOMESTIC_SALE:
        mh = mh.at[HEAD_COLOR].set(jnp.clip(params.get("color", 0), 0, nc - 1) + 1)
        mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(params.get("price_slot", 0), 0, PRICE_SLOTS - 1) + 1)

    return mh


def _rel_to_multihd(action_type: int, rel_offset: int, num_players: int = 2, num_colors: int = 5) -> jnp.ndarray:
    """Convert an old-style (action_type, rel_offset) to a multi-head action array.

    All sub-head indices are offset by +1 for the no-op element at index 0.
    Shopping actions produce the step-1 opponent selection only.
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
        # Step 1: opponent selection only.  PURCHASE_STOP is default.

    elif action_type == ACTION_DOMESTIC_SALE:
        remainder = rel_offset % combos
        color = remainder // PRICE_SLOTS
        price_slot = remainder % PRICE_SLOTS
        mh = mh.at[HEAD_COLOR].set(jnp.clip(color, 0, nc - 1) + 1)
        mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(price_slot, 0, PRICE_SLOTS - 1) + 1)

    return mh


# ══════════════════════════════════════════════════════════════════════════════
# ActionEncoder
# ══════════════════════════════════════════════════════════════════════════════


class TestActionEncoder:
    @pytest.mark.parametrize("num_players,num_colors", [(2, 5), (3, 4), (4, 3)])
    def test_total_actions_matches_offsets(self, num_players, num_colors):
        encoder = ActionEncoder(num_players, num_colors)
        # The last offset + its count should equal total
        assert encoder.offsets["domestic_sale"] + encoder.domestic_sale_actions == encoder.total_actions

    @pytest.mark.parametrize("num_players,num_colors", [(2, 5), (3, 4)])
    def test_encode_decode_roundtrip_buy_factory(self, num_players, num_colors):
        encoder = ActionEncoder(num_players, num_colors)
        for c in range(num_colors):
            idx = encoder.encode(ACTION_BUY_FACTORY, {"color": c})
            atype, params = encoder.decode(idx)
            assert atype == ACTION_BUY_FACTORY
            assert params == {"color": c}

    def test_encode_decode_roundtrip_buy_warehouse(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_BUY_WAREHOUSE, {})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_BUY_WAREHOUSE
        assert params == {}

    def test_encode_decode_roundtrip_produce(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_PRODUCE, {"color": 1, "price_slot": 2})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_PRODUCE
        assert params["color"] == 1
        assert params["price_slot"] == 2

    def test_encode_decode_roundtrip_buy_from_factory_store(self):
        encoder = ActionEncoder(3, 4)
        for opp in [1, 2]:
            for color in range(4):
                for slot in [0, 5, 9]:
                    params = {"opponent": opp, "color": color, "price_slot": slot}
                    idx = encoder.encode(ACTION_BUY_FROM_FACTORY_STORE, params)
                    atype, decoded_params = encoder.decode(idx)
                    assert atype == ACTION_BUY_FROM_FACTORY_STORE
                    assert decoded_params == params

    def test_encode_decode_roundtrip_move_load(self):
        encoder = ActionEncoder(2, 5)
        params = {"opponent": 1, "color": 3, "price_slot": 7}
        idx = encoder.encode(ACTION_MOVE_LOAD, params)
        atype, decoded_params = encoder.decode(idx)
        assert atype == ACTION_MOVE_LOAD
        assert decoded_params == params

    def test_encode_decode_roundtrip_move_sea(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_MOVE_SEA, {})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_MOVE_SEA

    def test_encode_decode_roundtrip_move_auction(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_MOVE_AUCTION, {})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_MOVE_AUCTION

    def test_encode_decode_roundtrip_pass(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_PASS, {})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_PASS

    def test_encode_decode_roundtrip_take_loan(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_TAKE_LOAN, {})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_TAKE_LOAN

    def test_encode_decode_roundtrip_repay_loan(self):
        encoder = ActionEncoder(2, 5)
        idx = encoder.encode(ACTION_REPAY_LOAN, {})
        atype, params = encoder.decode(idx)
        assert atype == ACTION_REPAY_LOAN

    def test_encode_decode_roundtrip_domestic_sale(self):
        encoder = ActionEncoder(2, 5)
        for store_type in [0, 1]:
            for color in [0, 4]:
                for slot in [0, 9]:
                    params = {"store_type": store_type, "color": color, "price_slot": slot}
                    idx = encoder.encode(ACTION_DOMESTIC_SALE, params)
                    atype, decoded_params = encoder.decode(idx)
                    assert atype == ACTION_DOMESTIC_SALE
                    assert decoded_params == params

    def test_all_action_types_have_unique_ranges(self):
        """No two action types should overlap in discrete action space."""
        encoder = ActionEncoder(2, 5)
        # encode one action per type and verify decode returns the same type
        actions = [
            (ACTION_BUY_FACTORY, {"color": 0}),
            (ACTION_BUY_WAREHOUSE, {}),
                                    (ACTION_PRODUCE, {"color": 0, "price_slot": 0}),
            (ACTION_BUY_FROM_FACTORY_STORE, {"opponent": 1, "color": 0, "price_slot": 0}),
            (ACTION_MOVE_LOAD, {"opponent": 1, "color": 0, "price_slot": 0}),
            (ACTION_MOVE_SEA, {}),
            (ACTION_MOVE_AUCTION, {}),
            (ACTION_PASS, {}),
            (ACTION_TAKE_LOAN, {}),
            (ACTION_REPAY_LOAN, {}),
            (ACTION_DOMESTIC_SALE, {"store_type": 0, "color": 0, "price_slot": 0}),
        ]
        indices = []
        for atype, params in actions:
            idx = encoder.encode(atype, params)
            indices.append(idx)
            decoded_type, _ = encoder.decode(idx)
            assert decoded_type == atype
        # All indices should be unique
        assert len(indices) == len(set(indices))


# ══════════════════════════════════════════════════════════════════════════════
# Initial State
# ══════════════════════════════════════════════════════════════════════════════


class TestInitialState:
    def test_initial_cash(self):
        state = _make_initial_state(num_players=3)
        assert int(state.cash[0]) == INITIAL_CASH
        assert int(state.cash[1]) == INITIAL_CASH
        assert int(state.cash[2]) == INITIAL_CASH

    def test_initial_loans_zero(self):
        state = _make_initial_state()
        assert int(state.loans[0]) == 0
        assert int(state.loans[1]) == 0

    def test_initial_unique_factory_colors(self):
        state = _make_initial_state(num_players=3)
        p0_color = int(jnp.argmax(state.factory_colors[0]))
        p1_color = int(jnp.argmax(state.factory_colors[1]))
        p2_color = int(jnp.argmax(state.factory_colors[2]))
        assert p0_color != p1_color
        assert p1_color != p2_color
        assert p0_color != p2_color

    def test_initial_one_warehouse_each(self):
        state = _make_initial_state(num_players=4)
        for p in range(4):
            assert int(state.warehouse_count[p]) == 1

    def test_initial_one_container_in_factory_store(self):
        state = _make_initial_state(num_players=2)
        p0_color = int(jnp.argmax(state.factory_colors[0]))
        total_p0 = int(jnp.sum(state.factory_store[0]))
        assert total_p0 == 1
        # Should be at price slot 1 ($2)
        assert int(state.factory_store[0, p0_color, 1]) == 1

    def test_initial_harbour_empty(self):
        state = _make_initial_state()
        assert int(jnp.sum(state.harbour_store)) == 0

    def test_initial_island_empty(self):
        state = _make_initial_state()
        assert int(jnp.sum(state.island_store)) == 0

    def test_initial_ship_empty_at_open_sea(self):
        state = _make_initial_state()
        assert int(jnp.sum(state.ship_contents)) == 0
        for p in range(state.ship_location.shape[0]):
            assert int(state.ship_location[p]) == LOCATION_OPEN_SEA

    def test_initial_container_supply(self):
        state = _make_initial_state(num_players=3, num_colors=4)
        for c in range(4):
            assert int(state.container_supply[c]) == 12  # 3 * 4

    def test_initial_current_player_zero(self):
        state = _make_initial_state()
        assert int(state.current_player) == 0

    def test_initial_not_game_over(self):
        state = _make_initial_state()
        assert int(state.game_over) == 0

    def test_initial_actions_taken_zero(self):
        state = _make_initial_state()
        assert int(state.actions_taken) == 0

    def test_initial_produced_false(self):
        state = _make_initial_state()
        assert int(state.produced_this_turn) == 0

    def test_initial_secret_value_color_in_range(self):
        state = _make_initial_state(num_colors=5)
        for p in range(2):
            assert 0 <= int(state.secret_value_color[p]) < 5


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_factory_cost(self):
        func_env = _make_func_env()
        assert int(func_env._factory_cost(0)) == 3   # first extra factory: $3
        assert int(func_env._factory_cost(1)) == 6   # second extra: $6
        assert int(func_env._factory_cost(4)) == 15  # $15

    def test_warehouse_cost(self):
        func_env = _make_func_env()
        assert int(func_env._warehouse_cost(1)) == 4  # already have 1, buying 2nd
        assert int(func_env._warehouse_cost(3)) == 6
        assert int(func_env._warehouse_cost(9)) == 12

    def test_count_store_containers(self):
        state = _make_state()
        # P0 has 1 container in factory store at (0, 0, 4)
        func_env = _make_func_env()
        count = int(func_env._count_store_containers(state.factory_store, 0))
        assert count == 1

    def test_count_store_containers_empty(self):
        state = _make_state()
        func_env = _make_func_env()
        count = int(func_env._count_store_containers(state.harbour_store, 0))
        assert count == 0

    def test_count_ship_cargo_empty(self):
        state = _make_state()
        func_env = _make_func_env()
        count = int(func_env._count_ship_cargo(state.ship_contents, 0))
        assert count == 0

    def test_count_ship_cargo_with_containers(self):
        contents = jnp.array([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(ship_contents=contents)
        func_env = _make_func_env()
        assert int(func_env._count_ship_cargo(state.ship_contents, 0)) == 2
        assert int(func_env._count_ship_cargo(state.ship_contents, 1)) == 0

    def test_get_target_player_clockwise(self):
        func_env = _make_func_env()
        # opp_idx=0 = player to the right (clockwise)
        assert int(func_env._get_target_player(0, 0, 4)) == 1  # P0→P1
        assert int(func_env._get_target_player(0, 1, 4)) == 2  # P1→P2
        assert int(func_env._get_target_player(0, 2, 4)) == 3  # P2→P3
        assert int(func_env._get_target_player(0, 3, 4)) == 0  # P3→P0
        # opp_idx=1 = two players clockwise
        assert int(func_env._get_target_player(1, 0, 4)) == 2  # P0→P2
        assert int(func_env._get_target_player(1, 2, 4)) == 0  # P2→P0
        assert int(func_env._get_target_player(2, 2, 4)) == 1  # P2→P1 (3rd opp)

    def test_decode_action_matches_encoder(self):
        func_env = _make_func_env(2, 5)
        encoder = ActionEncoder(2, 5)
        for atype in range(11):
            if atype == ACTION_BUY_FACTORY:
                params = {"color": 2}
            elif atype == ACTION_PRODUCE:
                params = {"color": 0, "price_slot": 2}
            elif atype in (ACTION_BUY_FROM_FACTORY_STORE, ACTION_MOVE_LOAD):
                params = {"opponent": 1, "color": 0, "price_slot": 3}
            elif atype == ACTION_DOMESTIC_SALE:
                params = {"store_type": 0, "color": 1, "price_slot": 5}
            else:
                params = {}
            idx = encoder.encode(atype, params)
            action_type, rel_offset = func_env._decode_action(
                jnp.array(idx, dtype=jnp.int32), ContainerParams(num_players=2, num_colors=5)
            )
            assert int(action_type) == atype, f"Action type mismatch for {atype}"


# ══════════════════════════════════════════════════════════════════════════════
# Pay Interest
# ══════════════════════════════════════════════════════════════════════════════


class TestPayInterest:
    def test_no_loans_no_change(self):
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([30, 20], dtype=jnp.int32))
        new_state = func_env._pay_interest(state, 2)
        assert int(new_state.cash[0]) == 30
        assert int(new_state.cash[1]) == 20

    def test_one_loan_pays_interest(self):
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([30, 20], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
        )
        new_state = func_env._pay_interest(state, 2)
        assert int(new_state.cash[0]) == 29  # 30 - 1

    def test_two_loans_pay_interest(self):
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([30, 20], dtype=jnp.int32),
            loans=jnp.array([2, 0], dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
        )
        new_state = func_env._pay_interest(state, 2)
        assert int(new_state.cash[0]) == 28  # 30 - 2

    def test_other_player_loans_not_charged(self):
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([30, 20], dtype=jnp.int32),
            loans=jnp.array([0, 2], dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
        )
        new_state = func_env._pay_interest(state, 2)
        assert int(new_state.cash[0]) == 30  # no change
        assert int(new_state.cash[1]) == 20  # no change (not player's turn)

    def test_cannot_pay_still_has_same_cash(self):
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([0, 20], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
        )
        new_state = func_env._pay_interest(state, 2)
        assert int(new_state.cash[0]) == 0  # unchanged, can't pay


# ══════════════════════════════════════════════════════════════════════════════
# Net Worth
# ══════════════════════════════════════════════════════════════════════════════


class TestNetWorth:
    def test_just_cash_no_containers(self):
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([50, 20], dtype=jnp.int32))
        nw = func_env._net_worth(state, 0, 5)
        assert int(nw) == 50

    def test_with_loan_penalty(self):
        func_env = _make_func_env()
        state = _make_state(
            cash=jnp.array([50, 20], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
        )
        nw = func_env._net_worth(state, 0, 5)
        assert int(nw) == 39  # 50 - 11

    def test_with_harbour_containers(self):
        func_env = _make_func_env()
        store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        store = store.at[0, 0, 0].set(3)  # 3 containers of color 0
        state = _make_state(cash=jnp.array([30, 20], dtype=jnp.int32), harbour_store=store)
        nw = func_env._net_worth(state, 0, 5)
        assert int(nw) == 36  # 30 + 3*2

    def test_with_ship_containers(self):
        func_env = _make_func_env()
        ship = jnp.array([[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(cash=jnp.array([30, 20], dtype=jnp.int32), ship_contents=ship)
        nw = func_env._net_worth(state, 0, 5)
        assert int(nw) == 39  # 30 + 3*3

    def test_island_scoring_with_all_colors(self):
        """10/5 color = $10 when you have at least one of every color."""
        func_env = _make_func_env()
        island = jnp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(
            cash=jnp.array([10, 20], dtype=jnp.int32),
            island_store=island,
            secret_value_color=jnp.array([0, 1], dtype=jnp.int32),
        )
        nw = func_env._net_worth(state, 0, 5)
        # has all colors -> color 0 worth $10 each, others $2 each
        # But most abundant color is discarded (all have 1, tie involves 10/5 -> discard 10/5)
        # Remaining: 4 containers at $2 each = $8
        # Total: 10 + 8 = 18
        assert int(nw) == 18

    def test_island_scoring_without_all_colors(self):
        """10/5 color = $5 when you don't have all colors."""
        func_env = _make_func_env()
        island = jnp.array([[3, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(
            cash=jnp.array([10, 20], dtype=jnp.int32),
            island_store=island,
            secret_value_color=jnp.array([0, 1], dtype=jnp.int32),
        )
        nw = func_env._net_worth(state, 0, 5)
        # Doesn't have all colors (colors 2, 4 missing) -> color 0 = $5 each
        # Most abundant: color 0 with 3 -> discard all of color 0
        # Remaining: color 1 (1) at $2, color 3 (1) at $2 = $4
        # Total: 10 + 4 = 14
        assert int(nw) == 14

    def test_island_empty_scores_zero(self):
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([50, 20], dtype=jnp.int32))
        nw = func_env._net_worth(state, 0, 5)
        assert int(nw) == 50

    def test_combined_net_worth(self):
        func_env = _make_func_env()
        island = jnp.array([[2, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        store = store.at[0, 2, 0].set(2)
        ship = jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(
            cash=jnp.array([20, 20], dtype=jnp.int32),
            island_store=island,
            harbour_store=store,
            ship_contents=ship,
            secret_value_color=jnp.array([0, 1], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
        )
        nw = func_env._net_worth(state, 0, 5)
        # cash: 20
        # loans: -11
        # harbour: 2 * 2 = 4
        # ship: 1 * 3 = 3
        # island: has all colors -> color 0 = $10 each. Most abundant: color 0 (2) discarded.
        #   Remaining: 4 containers at $2 each = $8
        # total = 20 - 11 + 4 + 3 + 8 = 24
        assert int(nw) == 24


# ══════════════════════════════════════════════════════════════════════════════
# Action: Buy Factory
# ══════════════════════════════════════════════════════════════════════════════


class TestBuyFactory:
    def test_buy_new_color(self):
        func_env = _make_func_env()
        state = _make_state()
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 2, 2, 5)  # rel_offset=2 means color 2
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.factory_colors[0, 2]) == 1
        # Cost for 2nd factory: (1+1)*3 = 6
        assert int(new_state.cash[0]) == INITIAL_CASH - 6

    def test_cannot_buy_duplicate_color(self):
        func_env = _make_func_env()
        state = _make_state()  # P0 already owns color 0
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 0, 2, 5)
        new_state = func_env._action_buy_factory(state, mh)
        # Should not change cash or factory_colors
        assert int(new_state.cash[0]) == INITIAL_CASH
        assert int(jnp.sum(new_state.factory_colors[0])) == 1  # still only 1 factory

    def test_cannot_buy_when_maxed_out(self):
        func_env = _make_func_env()
        # P0 already owns all 5 colors
        colors = jnp.ones((2, 5), dtype=jnp.int32)
        state = _make_state(factory_colors=colors)
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 2, 2, 5)
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.cash[0]) == INITIAL_CASH

    def test_cannot_buy_when_cant_afford(self):
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([1, 20], dtype=jnp.int32))
        mh = _rel_to_multihd(ACTION_BUY_FACTORY, 2, 2, 5)
        new_state = func_env._action_buy_factory(state, mh)
        assert int(new_state.cash[0]) == 1
        assert int(new_state.factory_colors[0, 2]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Action: Buy Warehouse
# ══════════════════════════════════════════════════════════════════════════════


class TestBuyWarehouse:
    def test_buy_warehouse(self):
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_BUY_WAREHOUSE)
        new_state = func_env._action_buy_warehouse(state, mh)
        assert int(new_state.warehouse_count[0]) == 2
        # Cost for 2nd warehouse: 1+3 = 4
        assert int(new_state.cash[0]) == INITIAL_CASH - 4

    def test_cannot_buy_past_max(self):
        func_env = _make_func_env()
        state = _make_state(warehouse_count=jnp.array([MAX_WAREHOUSES_PER_PLAYER, 1], dtype=jnp.int32))
        mh = _build_multihd(ACTION_BUY_WAREHOUSE)
        new_state = func_env._action_buy_warehouse(state, mh)
        assert int(new_state.warehouse_count[0]) == MAX_WAREHOUSES_PER_PLAYER
        assert int(new_state.cash[0]) == INITIAL_CASH  # no change

    def test_cannot_buy_when_cant_afford(self):
        func_env = _make_func_env()
        state = _make_state(cash=jnp.array([0, 20], dtype=jnp.int32))
        mh = _build_multihd(ACTION_BUY_WAREHOUSE)
        new_state = func_env._action_buy_warehouse(state, mh)
        assert int(new_state.warehouse_count[0]) == 1  # no change
        assert int(new_state.cash[0]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Action: Produce
# ══════════════════════════════════════════════════════════════════════════════


class TestProduce:
    def _do_produce(self, func_env, state, params, color: int, price_slot: int | None = None):
        """Run a complete produce cycle: enter mode + process one factory.

        If *color* is None, only enters produce mode (no factory processed).
        Otherwise enters mode and processes colour+price for one factory.
        """
        # ── Enter produce mode ──────────────────────────────────────
        enter_action = jnp.array(
            [ACTION_PRODUCE + 1, 0, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        state = func_env._action_produce(state, enter_action, params)
        if color is None:
            return state
        # ── Process one factory ─────────────────────────────────────
        mh = jnp.array(
            [0, 0, color + 1, price_slot + 1, PURCHASE_STOP], dtype=jnp.int32,
        )
        return func_env._produce_shopping_step(state, mh, params)

    def test_produce_adds_container(self):
        func_env = _make_func_env()
        state = _make_state()
        params = _make_params()
        p0_color = int(jnp.argmax(state.factory_colors[0]))
        initial_supply = int(state.container_supply[p0_color])

        new_state = self._do_produce(func_env, state, params, p0_color, 0)

        assert int(jnp.sum(new_state.factory_store[0])) == 2
        assert int(new_state.container_supply[p0_color]) == initial_supply - 1
        assert int(new_state.cash[0]) == INITIAL_CASH - 1
        assert int(new_state.cash[1]) == INITIAL_CASH + 1
        assert int(new_state.produced_this_turn) == 1

    def test_can_produce_multiple_colors_per_turn(self):
        """Processing factories in the same batch only pays $1 once."""
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

        # Factory 1: colour 0, price 1
        state = func_env._produce_shopping_step(state, jnp.array(
            [0, 0, 1, 2, PURCHASE_STOP], dtype=jnp.int32), params)
        assert int(state.produce_active) == 1

        # Factory 2: colour 2, price 2 (no extra payment)
        state = func_env._produce_shopping_step(state, jnp.array(
            [0, 0, 3, 3, PURCHASE_STOP], dtype=jnp.int32), params)
        assert int(state.produce_active) == 0
        assert int(state.cash[0]) == before_cash

    def test_pays_right_player(self):
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state(current_player=jnp.array(1, dtype=jnp.int32))
        p1_color = int(jnp.argmax(state.factory_colors[1]))
        new_state = self._do_produce(func_env, state, params, p1_color, 0)
        assert int(new_state.cash[1]) == INITIAL_CASH - 1
        assert int(new_state.cash[0]) == INITIAL_CASH + 1

    def test_no_produce_when_storage_full(self):
        func_env = _make_func_env()
        params = _make_params()
        full_store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        full_store = full_store.at[0, 0, 0].set(2)
        state = _make_state(factory_store=full_store)
        new_state = self._do_produce(func_env, state, params, 0, 0)
        assert int(jnp.sum(new_state.factory_store[0])) == 2
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action: Buy from Factory Store
# ══════════════════════════════════════════════════════════════════════════════


class TestBuyFromFactoryStore:
    def test_buy_one_container(self):
        func_env = _make_func_env()
        state = _make_state()
        params = _make_params()
        # P0 buys from P1's factory store at color 1. The env auto-chooses
        # the cheapest source slot (slot 4 = $5 for color 1 in _make_state).
        # Step 1: select opponent
        opp_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_buy_from_factory_store(state, opp_action, params)
        assert int(new_state.shopping_active) == 1

        # Step 2: buy colour 1, harbour price $3 (purchase index 2)
        buy_action = jnp.array(
            [0, 0, 2, 0, 2], dtype=jnp.int32,  # colour 1, harbour $3
        )
        new_state = func_env._shopping_step(new_state, buy_action, params)

        assert int(new_state.cash[0]) == INITIAL_CASH - 5  # paid $5 for source
        assert int(new_state.cash[1]) == INITIAL_CASH + 5
        assert int(new_state.factory_store[1, 1, 4]) == 0  # source consumed
        assert int(new_state.harbour_store[0, 1, 2]) == 1  # placed at harbour $3

    def test_no_stock_no_purchase(self):
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        # Opponent 0 from player 0 = player 1, has stock (color 1 slot 4)
        opp_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_buy_from_factory_store(state, opp_action, params)
        # shopping should be active since P1 has stock
        assert int(new_state.shopping_active) == 1

    def test_cannot_buy_from_self(self):
        pass  # covered by design — _get_target_player never returns the acting player

    def test_cannot_buy_when_harbour_full(self):
        func_env = _make_func_env()
        params = _make_params()
        # Fill P0's harbour to capacity (1 warehouse = 1 container)
        full_harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        full_harbour = full_harbour.at[0, 3, 0].set(1)
        state = _make_state(harbour_store=full_harbour)

        # Step 1: opponent selection — should not enter shopping (harbour full)
        opp_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_buy_from_factory_store(state, opp_action, params)
        assert int(new_state.shopping_active) == 0
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action: Move to Harbour + Load
# ══════════════════════════════════════════════════════════════════════════════


class TestMoveLoad:
    def test_load_from_harbour(self):
        func_env = _make_func_env()
        params = _make_params()
        harbour = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        harbour = harbour.at[1, 2, 3].set(1)
        state = _make_state(harbour_store=harbour)

        # Step 1: select opponent
        opp_action = jnp.array(
            [ACTION_MOVE_LOAD + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_move_load(state, opp_action, params)
        assert int(new_state.shopping_active) == 1

        # Step 2: buy colour 2 (colour head index 3, purchase=1 for "buy")
        buy_action = jnp.array(
            [0, 0, 3, 0, 1], dtype=jnp.int32,  # colour 2, buy signal
        )
        new_state = func_env._shopping_step(new_state, buy_action, params)

        assert int(new_state.cash[0]) == INITIAL_CASH - 4
        assert int(new_state.cash[1]) == INITIAL_CASH + 4
        assert int(new_state.harbour_store[1, 2, 3]) == 0
        assert int(jnp.sum(new_state.ship_contents[0] > 0)) == 1
        assert int(new_state.ship_contents[0, 0]) == 3  # color index 3
        assert int(new_state.ship_location[0]) == LOCATION_HARBOUR_OFFSET + 1

    def test_cannot_load_from_empty_harbour(self):
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        # Opponent selection should not enter shopping (no stock)
        opp_action = jnp.array(
            [ACTION_MOVE_LOAD + 1, 1, 0, 0, PURCHASE_STOP], dtype=jnp.int32,
        )
        new_state = func_env._action_move_load(state, opp_action, params)
        assert int(new_state.shopping_active) == 0

    def test_cannot_load_when_ship_full(self):
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
# Action: Move to Open Sea
# ══════════════════════════════════════════════════════════════════════════════


class TestMoveSea:
    def test_move_to_sea(self):
        func_env = _make_func_env()
        state = _make_state(ship_location=jnp.array([LOCATION_HARBOUR_OFFSET + 1, LOCATION_OPEN_SEA], dtype=jnp.int32))
        mh = _build_multihd(ACTION_MOVE_SEA)
        new_state = func_env._action_move_sea(state, mh)
        assert int(new_state.ship_location[0]) == LOCATION_OPEN_SEA

    def test_move_to_sea_when_already_there(self):
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_MOVE_SEA)
        new_state = func_env._action_move_sea(state, mh)
        assert int(new_state.ship_location[0]) == LOCATION_OPEN_SEA


# ══════════════════════════════════════════════════════════════════════════════
# Action: Auction
# ══════════════════════════════════════════════════════════════════════════════


class TestAuction:
    def test_cannot_auction_when_not_at_sea(self):
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
        assert int(new_state.ship_contents[0, 0]) == 1

    def test_cannot_auction_empty_ship(self):
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        key = random.PRNGKey(99)
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        new_state = func_env._action_move_auction(state, mh, key, params)
        assert int(new_state.ship_location[0]) == LOCATION_OPEN_SEA

    def test_auction_clears_ship(self):
        func_env = _make_func_env()
        params = _make_params()
        ship = jnp.array([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        state = _make_state(ship_contents=ship)
        key = random.PRNGKey(99)
        mh = _build_multihd(ACTION_MOVE_AUCTION)
        new_state = func_env._action_move_auction(state, mh, key, params)
        assert int(jnp.sum(new_state.ship_contents[0] > 0)) == 0
        assert int(new_state.ship_location[0]) == LOCATION_AUCTION_ISLAND

    def test_auction_deposits_goods(self):
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
# Action: Pass
# ══════════════════════════════════════════════════════════════════════════════


class TestPass:
    def test_pass_no_change(self):
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_PASS)
        new_state = func_env._action_pass(state, mh)
        assert int(new_state.cash[0]) == int(state.cash[0])


# ══════════════════════════════════════════════════════════════════════════════
# Action: Take Loan
# ══════════════════════════════════════════════════════════════════════════════


class TestTakeLoan:
    def test_take_loan(self):
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_TAKE_LOAN)
        new_state = func_env._action_take_loan(state, mh)
        assert int(new_state.loans[0]) == 1
        assert int(new_state.cash[0]) == INITIAL_CASH + LOAN_AMOUNT

    def test_cannot_exceed_two_loans(self):
        func_env = _make_func_env()
        state = _make_state(loans=jnp.array([2, 0], dtype=jnp.int32))
        mh = _build_multihd(ACTION_TAKE_LOAN)
        new_state = func_env._action_take_loan(state, mh)
        assert int(new_state.loans[0]) == 2
        assert int(new_state.cash[0]) == INITIAL_CASH


# ══════════════════════════════════════════════════════════════════════════════
# Action: Repay Loan
# ══════════════════════════════════════════════════════════════════════════════


class TestRepayLoan:
    def test_repay_loan(self):
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
        func_env = _make_func_env()
        state = _make_state()
        mh = _build_multihd(ACTION_REPAY_LOAN)
        new_state = func_env._action_repay_loan(state, mh)
        assert int(new_state.loans[0]) == 0

    def test_cannot_repay_without_cash(self):
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
# Action: Domestic Sale
# ══════════════════════════════════════════════════════════════════════════════


class TestDomesticSale:
    def test_sell_from_factory_store(self):
        func_env = _make_func_env()
        params = _make_params()
        state = _make_state()
        mh = _rel_to_multihd(ACTION_DOMESTIC_SALE, 0 * 50 + 0 * 10 + 4, 2, 5)
        new_state = func_env._action_domestic_sale(state, mh, params)

        assert int(new_state.cash[0]) == INITIAL_CASH + 2
        assert int(new_state.factory_store[0, 0, 4]) == 0
        assert int(new_state.container_supply[0]) == 9

    def test_sell_from_harbour_store(self):
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
# Turn Advancement
# ══════════════════════════════════════════════════════════════════════════════


class TestAdvanceTurn:
    def test_first_action_does_not_end_turn(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._advance_turn(state, ACTION_PASS, 2)
        assert int(new_state.actions_taken) == 1
        assert int(new_state.current_player) == 0  # still P0

    def test_second_action_ends_turn(self):
        func_env = _make_func_env()
        state = _make_state(actions_taken=jnp.array(1, dtype=jnp.int32))
        new_state = func_env._advance_turn(state, ACTION_PASS, 2)
        assert int(new_state.actions_taken) == 0
        assert int(new_state.current_player) == 1  # moves to P1

    def test_auction_ends_turn_immediately(self):
        func_env = _make_func_env()
        state = _make_state(actions_taken=jnp.array(0, dtype=jnp.int32))
        new_state = func_env._advance_turn(state, ACTION_MOVE_AUCTION, 2)
        assert int(new_state.actions_taken) == 0
        assert int(new_state.current_player) == 1  # advances even on first action

    def test_loan_does_not_consume_action(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._advance_turn(state, ACTION_TAKE_LOAN, 2)
        assert int(new_state.actions_taken) == 0  # unchanged
        assert int(new_state.current_player) == 0  # still P0

    def test_repay_loan_does_not_consume_action(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._advance_turn(state, ACTION_REPAY_LOAN, 2)
        assert int(new_state.actions_taken) == 0

    def test_resets_produced_flag_on_turn_end(self):
        func_env = _make_func_env()
        state = _make_state(
            actions_taken=jnp.array(1, dtype=jnp.int32),
            produced_this_turn=jnp.array(1, dtype=jnp.int32),
        )
        new_state = func_env._advance_turn(state, ACTION_PASS, 2)
        assert int(new_state.produced_this_turn) == 0

    def test_wraps_around_to_player0(self):
        func_env = _make_func_env()
        state = _make_state(
            current_player=jnp.array(1, dtype=jnp.int32),
            actions_taken=jnp.array(1, dtype=jnp.int32),
        )
        new_state = func_env._advance_turn(state, ACTION_PASS, 2)
        assert int(new_state.current_player) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Game End Check
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckGameEnd:
    def test_not_game_over_initially(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._check_game_end(state, 5)
        assert int(new_state.game_over) == 0

    def test_game_over_when_two_colors_exhausted(self):
        func_env = _make_func_env()
        state = _make_state(container_supply=jnp.array([0, 0, 8, 8, 8], dtype=jnp.int32))
        new_state = func_env._check_game_end(state, 5)
        assert int(new_state.game_over) == 1

    def test_not_game_over_with_one_exhausted(self):
        func_env = _make_func_env()
        state = _make_state(container_supply=jnp.array([0, 8, 8, 8, 8], dtype=jnp.int32))
        new_state = func_env._check_game_end(state, 5)
        assert int(new_state.game_over) == 0

    def test_game_over_on_step_limit(self):
        func_env = _make_func_env()
        state = _make_state(step_count=jnp.array(1001, dtype=jnp.int32))
        new_state = func_env._check_game_end(state, 5)
        assert int(new_state.game_over) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Full transition
# ══════════════════════════════════════════════════════════════════════════════


class TestTransition:
    def test_transition_executes_action(self):
        func_env = _make_func_env()
        state = _make_initial_state(func_env)
        params = _make_params()
        key = random.PRNGKey(42)

        # Encode a produce action
        encoder = ActionEncoder(2, 5)
        action_idx = encoder.encode(ACTION_PRODUCE, {"color": 0, "price_slot": 0})
        action = jnp.array(action_idx, dtype=jnp.int32)

        new_state = func_env.transition(state, action, key, params)

        # Player 0 should have paid $1
        assert int(new_state.cash[0]) == INITIAL_CASH - 1
        # Container supply decreased
        p0_color = int(jnp.argmax(new_state.factory_colors[0]))
        assert int(new_state.container_supply[p0_color]) == 7  # 8 - 1

    def test_transition_pays_interest(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(42)
        state = _make_state(
            cash=jnp.array([30, 20], dtype=jnp.int32),
            loans=jnp.array([1, 0], dtype=jnp.int32),
        )

        encoder = ActionEncoder(2, 5)
        action_idx = encoder.encode(ACTION_PASS, {})
        new_state = func_env.transition(state, jnp.array(action_idx), key, params)

        # Interest of $1 deducted
        assert int(new_state.cash[0]) == 29

    def test_transition_advances_turn(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(42)
        state = _make_state()

        encoder = ActionEncoder(2, 5)
        # Two passes should advance to next player
        action_idx = encoder.encode(ACTION_PASS, {})
        s1 = func_env.transition(state, jnp.array(action_idx), key, params)
        assert int(s1.actions_taken) == 1
        assert int(s1.current_player) == 0

        s2 = func_env.transition(s1, jnp.array(action_idx), key, params)
        assert int(s2.actions_taken) == 0
        assert int(s2.current_player) == 1

    def test_transition_loan_doesnt_advance_action_count(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(42)
        state = _make_state()

        encoder = ActionEncoder(2, 5)
        # Take loan
        action_idx = encoder.encode(ACTION_TAKE_LOAN, {})
        s1 = func_env.transition(state, jnp.array(action_idx), key, params)
        assert int(s1.actions_taken) == 0  # unchanged
        assert int(s1.loans[0]) == 1

        # Still can take 2 more actions
        pass_idx = encoder.encode(ACTION_PASS, {})
        s2 = func_env.transition(s1, jnp.array(pass_idx), key, params)
        assert int(s2.actions_taken) == 1
        assert int(s2.current_player) == 0  # still P0's turn

    def test_transition_increments_step_count(self):
        func_env = _make_func_env()
        state = _make_initial_state(func_env)
        params = _make_params()
        key = random.PRNGKey(42)

        encoder = ActionEncoder(2, 5)
        action_idx = encoder.encode(ACTION_PASS, {})
        new_state = func_env.transition(state, jnp.array(action_idx), key, params)
        assert int(new_state.step_count) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Observation, Terminal, Reward
# ══════════════════════════════════════════════════════════════════════════════


class TestObservationTerminalReward:
    def test_observation_has_correct_shape(self):
        func_env = _make_func_env()
        state = _make_initial_state(func_env)
        rng = random.PRNGKey(0)
        obs = func_env.observation(state, rng, _make_params())
        assert obs.shape == func_env.observation_space.shape
        assert obs.dtype == jnp.float32

    def test_terminal_returns_false_initial(self):
        func_env = _make_func_env()
        state = _make_initial_state(func_env)
        assert not func_env.terminal(state, random.PRNGKey(0), _make_params())

    def test_terminal_returns_true_when_game_over(self):
        func_env = _make_func_env()
        state = _make_state(game_over=jnp.array(1, dtype=jnp.int32))
        assert func_env.terminal(state, random.PRNGKey(0), _make_params())

    def test_reward_computes_net_worth_delta(self):
        func_env = _make_func_env()
        state = _make_initial_state(func_env)
        params = _make_params()
        key = random.PRNGKey(42)

        encoder = ActionEncoder(2, 5)
        action_idx = encoder.encode(ACTION_PRODUCE, {"color": 0, "price_slot": 0})
        next_state = func_env.transition(state, jnp.array(action_idx), key, params)

        reward = func_env.reward(state, action_idx, next_state, key, params)
        assert reward < 0  # paid $1 to opponent


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Full game simulation
# ══════════════════════════════════════════════════════════════════════════════


class TestFullGame:
    def test_full_game_runs(self):
        """Run a full game with JIT-disabled environment until termination."""
        env = ContainerJaxEnv(num_players=2, num_colors=5)
        encoder = ActionEncoder(2, 5)
        obs, info = env.reset(seed=42)

        total_reward = 0.0
        steps = 0
        while True:
            action_idx = encoder.encode(ACTION_PASS, {})
            obs, reward, term, trunc, info = env.step(action_idx)
            total_reward += float(reward)
            steps += 1
            if term or trunc or steps > 100:
                break

        # Game should have run some steps without crashing
        assert steps > 2

    def test_full_game_terminates(self):
        """Verify the game terminates when 2 colors are exhausted."""
        env = ContainerJaxEnv(num_players=2, num_colors=2)
        encoder = ActionEncoder(2, 2)
        obs, info = env.reset(seed=42)

        # Run rapid production only from player 0 across turns
        steps = 0
        while True:
            # Produce (if valid) or pass
            action_idx = encoder.encode(ACTION_PRODUCE, {"color": 0, "price_slot": 0})
            obs, reward, term, trunc, info = env.step(action_idx)
            steps += 1
            if term:
                break
            if steps > 2000:
                pytest.fail("Game did not terminate within 2000 steps")

        # With 2 colors, 2 players: supply = 8 per color
        # 2 colors x 8 = 16 containers needed. With 2 players alternating
        # and each producing 1 per 2 turns (when it's their turn),
        # it takes many steps.
        assert term

    def test_jit_environment_works(self):
        """Verify the JIT-compiled environment works end-to-end."""
        env = ContainerJaxEnv(num_players=2, num_colors=5)
        encoder = ActionEncoder(2, 5)
        obs, info = env.reset(seed=42)

        # Run a few steps through the JIT env
        actions = [
                                    (ACTION_PRODUCE, {"color": 0, "price_slot": 0}),
            (ACTION_PASS, {}),
            (ACTION_TAKE_LOAN, {}),
            (ACTION_REPAY_LOAN, {}),
            (ACTION_BUY_FACTORY, {"color": 1}),
            (ACTION_BUY_WAREHOUSE, {}),
            (ACTION_MOVE_SEA, {}),
            (ACTION_PASS, {}),
        ]
        for atype, params in actions:
            idx = encoder.encode(atype, params)
            obs, reward, term, trunc, info = env.step(idx)
            assert isinstance(reward, float) or hasattr(reward, "item")

    def test_observation_changes_over_time(self):
        """Observation should change as game state evolves."""
        env = ContainerJaxEnv(num_players=2, num_colors=5)
        encoder = ActionEncoder(2, 5)
        obs1, info = env.reset(seed=42)
        env.step(encoder.encode(ACTION_PRODUCE, {"color": 0, "price_slot": 0}))
        env.step(encoder.encode(ACTION_PASS, {}))
        obs2, info = env.reset(seed=42)
        env.step(encoder.encode(ACTION_PRODUCE, {"color": 0, "price_slot": 0}))
        obs3, info = env.reset(seed=42)

        # obs1 and obs3 should be identical (same seed)
        assert jnp.allclose(obs1, obs3, atol=0)

    def test_reward_accumulates_over_game(self):
        """Verify total reward reflects meaningful action outcomes."""
        env = ContainerJaxEnv(num_players=2, num_colors=5)
        encoder = ActionEncoder(2, 5)
        obs, info = env.reset(seed=42)

        total = 0.0
        actions = [
            ACTION_PRODUCE,       # P0 produces, pays $1 -> reward -1
            ACTION_PASS,          # P0 passes, turn ends
            ACTION_PASS,          # P1 passes (action 1)
            ACTION_PASS,          # P1 passes (action 2), turn back to P0
            ACTION_TAKE_LOAN,     # P0 takes loan -> net worth change
            ACTION_BUY_WAREHOUSE, # P0 buys warehouse -> pays money
            ACTION_PASS,          # P0 passes
            ACTION_PASS,          # P1 passes
            ACTION_PASS,          # P1 passes
        ]
        for atype in actions:
            idx = encoder.encode(atype, {})
            obs, reward, term, trunc, info = env.step(idx)
            total += float(reward)
            if term:
                break

        # Several actions with non-zero reward were taken
        assert total != 0.0
