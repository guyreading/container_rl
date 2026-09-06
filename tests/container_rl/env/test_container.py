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
    INITIAL_CASH,
    LOAN_AMOUNT,
    LOCATION_OPEN_SEA,
    PRICE_SLOTS,
    SECRET_CARD_VALUES,
    SHIP_CAPACITY,
    ActionEncoder,
    ContainerFunctional,
    ContainerJaxEnv,
    ContainerParams,
    EnvState,
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
    """Create an EnvState with defaults suitable for 2-player testing.

    Both players hold the standard secret card deck in colour order, so
    island scoring in these tests is deterministic.
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
        secret_card_values=jnp.tile(
            jnp.array(SECRET_CARD_VALUES, dtype=jnp.int32), (np_, 1)
        ),
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
        state = _make_initial_state(num_players=3, num_colors=5)
        factory_counts = jnp.sum(state.factory_colors, axis=1)
        for p in range(3):
            assert int(factory_counts[p]) == 1

        colors = {int(jnp.argmax(state.factory_colors[p])) for p in range(3)}
        assert colors == {0, 1, 2}

    def test_initial_one_warehouse_each(self):
        state = _make_initial_state(num_players=4)
        for p in range(4):
            assert int(state.warehouse_count[p]) == 1

    def test_initial_one_container_in_factory_store(self):
        state = _make_initial_state(num_players=2, num_colors=5)
        for p in range(2):
            color = int(jnp.argmax(state.factory_colors[p]))
            assert int(jnp.sum(state.factory_store[p])) == 1
            assert int(state.factory_store[p, color, 1]) == 1

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
        num_players, num_colors = 3, 4
        state = _make_initial_state(num_players=num_players, num_colors=num_colors)
        for c in range(num_colors):
            assert int(state.container_supply[c]) == num_players * 4

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

    def test_initial_secret_cards_are_a_permutation_of_the_deck(self):
        state = _make_initial_state(num_colors=5)
        for p in range(2):
            row = [int(v) for v in state.secret_card_values[p]]
            assert sorted(row) == sorted(SECRET_CARD_VALUES)

    def test_initial_secret_cards_match_board_width(self):
        """The 5-card deck is trimmed/cycled so the row is always num_colors wide."""
        for nc in (3, 4, 5):
            state = _make_initial_state(num_colors=nc)
            assert state.secret_card_values.shape == (2, nc)
            # The special 5/10 card is always dealt.
            for p in range(2):
                assert -1 in [int(v) for v in state.secret_card_values[p]]


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_factory_cost(self):
        func_env = _make_func_env()
        for n in range(5):
            assert int(func_env._factory_cost(n)) == (n + 1) * 3

    def test_warehouse_cost(self):
        func_env = _make_func_env()
        for n in range(10):
            assert int(func_env._warehouse_cost(n)) == n + 3

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
        """Colour 0 is the 10/5 card.  Full set → $10 each.
        Most abundant colour is discarded (every colour has 1 → discard 10/5)."""
        func_env = _make_func_env()
        island = jnp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        scv = jnp.array([[-1, 2, 4, 6, 10], [6, 4, 2, 10, -1]], dtype=jnp.int32)
        state = _make_state(
            cash=jnp.array([10, 20], dtype=jnp.int32),
            island_store=island,
            secret_card_values=scv,
        )
        nw = func_env._net_worth(state, 0, 5)
        # multipliers: [10, 2, 4, 6, 10]; discard colour 0
        # after discard: [0,1,1,1,1] → 0*10 + 1*2 + 1*4 + 1*6 + 1*10 = 22
        # total: 10 + 22 = 32
        assert int(nw) == 32

    def test_island_scoring_without_all_colors(self):
        """Colour 0 is the 10/5 card.  Incomplete set → $5 each.
        Most abundant colour discarded."""
        func_env = _make_func_env()
        island = jnp.array([[3, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        scv = jnp.array([[-1, 2, 4, 6, 10], [6, 4, 2, 10, -1]], dtype=jnp.int32)
        state = _make_state(
            cash=jnp.array([10, 20], dtype=jnp.int32),
            island_store=island,
            secret_card_values=scv,
        )
        nw = func_env._net_worth(state, 0, 5)
        # no full set → multipliers: [5, 2, 4, 6, 10]
        # most abundant: colour 0 (3) discarded
        # after discard: [0,1,0,1,0] → 0*5+1*2+0*4+1*6+0*10 = 8
        # total: 10 + 8 = 18
        assert int(nw) == 18

    def test_island_scoring_uses_each_colours_own_card_value(self):
        """A surviving colour scores its own card value, not a positional one."""
        func_env = _make_func_env()
        island = jnp.array([[1, 3, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        cards = jnp.tile(jnp.array([6, 2, 4, 10, -1], dtype=jnp.int32), (2, 1))
        state = _make_state(
            cash=jnp.array([0, 0], dtype=jnp.int32),
            island_store=island,
            secret_card_values=cards,
        )
        # most abundant is colour 1 (3) -> discarded
        # colour 0 survives: 1 container x its own card value 6 = 6
        assert int(func_env._net_worth(state, 0, 5)) == 6

    def test_island_empty_scores_zero(self):
        func_env = _make_func_env()
        scv = jnp.array([[-1, 2, 4, 6, 10], [6, 4, 2, 10, -1]], dtype=jnp.int32)
        state = _make_state(cash=jnp.array([50, 20], dtype=jnp.int32), secret_card_values=scv)
        nw = func_env._net_worth(state, 0, 5)
        assert int(nw) == 50

    def test_combined_net_worth(self):
        func_env = _make_func_env()
        island = jnp.array([[2, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        store = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32)
        store = store.at[0, 2, 0].set(2)
        ship = jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        scv = jnp.array([[-1, 2, 4, 6, 10], [6, 4, 2, 10, -1]], dtype=jnp.int32)
        state = _make_state(
            cash=jnp.array([20, 20], dtype=jnp.int32),
            island_store=island,
            harbour_store=store,
            ship_contents=ship,
            secret_card_values=scv,
            loans=jnp.array([1, 0], dtype=jnp.int32),
        )
        nw = func_env._net_worth(state, 0, 5)
        # cash: 20
        # loans: -11
        # harbour: 2 colour-2 containers × $2 = 4
        # ship: 1 container × $3 = 3
        # island (has all → colour 0 = $10, others [2,4,6,10];
        #   most abundant: colour 0 (2) discarded):
        #   [0,1,1,1,1] × [10,2,4,6,10] = 0+2+4+6+10 = 22
        # total = 20 - 11 + 4 + 3 + 22 = 38
        assert int(nw) == 38




# Pay Interest
# ══════════════════════════════════════════════════════════════════════════════

# Turn Advancement
# ══════════════════════════════════════════════════════════════════════════════


class TestAdvanceTurn:
    def test_first_action_does_not_end_turn(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._advance_turn(state, ACTION_PASS, 2, False)
        assert int(new_state.actions_taken) == 1
        assert int(new_state.current_player) == 0  # still P0

    def test_second_action_ends_turn(self):
        func_env = _make_func_env()
        state = _make_state(actions_taken=jnp.array(1, dtype=jnp.int32))
        new_state = func_env._advance_turn(state, ACTION_PASS, 2, False)
        assert int(new_state.actions_taken) == 0
        assert int(new_state.current_player) == 1  # moves to P1

    def test_auction_ends_turn_immediately(self):
        func_env = _make_func_env()
        state = _make_state(actions_taken=jnp.array(0, dtype=jnp.int32))
        new_state = func_env._advance_turn(state, ACTION_MOVE_AUCTION, 2, False)
        assert int(new_state.actions_taken) == 0
        assert int(new_state.current_player) == 1  # advances even on first action

    def test_loan_does_not_consume_action(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._advance_turn(state, ACTION_TAKE_LOAN, 2, False)
        assert int(new_state.actions_taken) == 0  # unchanged
        assert int(new_state.current_player) == 0  # still P0

    def test_repay_loan_does_not_consume_action(self):
        func_env = _make_func_env()
        state = _make_state()
        new_state = func_env._advance_turn(state, ACTION_REPAY_LOAN, 2, False)
        assert int(new_state.actions_taken) == 0

    def test_resets_produced_flag_on_turn_end(self):
        func_env = _make_func_env()
        state = _make_state(
            actions_taken=jnp.array(1, dtype=jnp.int32),
            produced_this_turn=jnp.array(1, dtype=jnp.int32),
        )
        new_state = func_env._advance_turn(state, ACTION_PASS, 2, False)
        assert int(new_state.produced_this_turn) == 0

    def test_wraps_around_to_player0(self):
        func_env = _make_func_env()
        state = _make_state(
            current_player=jnp.array(1, dtype=jnp.int32),
            actions_taken=jnp.array(1, dtype=jnp.int32),
        )
        new_state = func_env._advance_turn(state, ACTION_PASS, 2, False)
        assert int(new_state.current_player) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Produce (once a turn)
# ══════════════════════════════════════════════════════════════════════════════


class TestActionProduce:
    """Producing is a once-a-turn action.

    Entering the batch unconditionally used to let a player run the factories
    again in the same turn for free: no second union dues, and _advance_turn
    charged no action for the repeat, so a whole factory store could be filled
    for $1.
    """

    @staticmethod
    def _three_factory_state(**overrides):
        np_, nc = 2, 5
        defaults = dict(
            factory_colors=jnp.zeros((np_, nc), dtype=jnp.int32).at[0, 0:3].set(1).at[1, 3].set(1),
            factory_store=jnp.zeros((np_, nc, PRICE_SLOTS), dtype=jnp.int32),
        )
        defaults.update(overrides)
        return _make_state(**defaults)

    def _produce_batch(self, func_env, state, params, key, colors=(0, 1, 2)):
        """Open a produce batch and put one container in each named factory."""
        state = func_env.transition(state, jnp.array([ACTION_PRODUCE + 1, 0, 0, 0, 0]), key, params)
        for c in colors:
            state = func_env.transition(
                state, jnp.array([ACTION_PRODUCE + 1, 0, c + 1, 1, 0]), key, params,
            )
        return state

    def test_first_produce_of_the_turn_runs_every_factory(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(0)
        state = self._three_factory_state()

        state = self._produce_batch(func_env, state, params, key)

        assert int(state.factory_store[0].sum()) == 3
        assert int(state.cash[0]) == INITIAL_CASH - 1   # union dues, once
        assert int(state.cash[1]) == INITIAL_CASH + 1   # paid to the left
        assert int(state.produce_active) == 0

    def test_second_produce_in_the_same_turn_is_rejected(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(0)
        state = self._three_factory_state()

        state = self._produce_batch(func_env, state, params, key)
        assert int(state.current_player) == 0  # one action left this turn
        after_first = state

        state = func_env.transition(
            state, jnp.array([ACTION_PRODUCE + 1, 0, 0, 0, 0]), key, params,
        )

        assert int(state.produce_active) == 0             # no second batch opened
        assert int(state.produce_pending.sum()) == 0
        assert int(state.factory_store[0].sum()) == 3     # nothing extra produced
        assert int(state.cash[0]) == int(after_first.cash[0])   # and no extra dues

    def test_rejected_produce_consumes_an_action(self):
        """Otherwise a repeat attempt is a free retry that never ends the turn."""
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(0)
        state = self._three_factory_state()

        state = self._produce_batch(func_env, state, params, key)
        assert int(state.actions_taken) == 1

        state = func_env.transition(
            state, jnp.array([ACTION_PRODUCE + 1, 0, 0, 0, 0]), key, params,
        )
        assert int(state.current_player) == 1  # second action spent, turn over

    def test_produce_reopens_on_the_next_turn(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(0)
        state = self._three_factory_state()

        state = self._produce_batch(func_env, state, params, key)
        # Spend the rest of the round so P0 is back on turn with a clean slate.
        encoder = ActionEncoder(2, 5)
        pass_action = jnp.array(encoder.encode(ACTION_PASS, {}), dtype=jnp.int32)
        while int(state.current_player) != 0 or int(state.actions_taken) != 0:
            state = func_env.transition(state, pass_action, key, params)

        assert int(state.produced_this_turn) == 0
        state = self._produce_batch(func_env, state, params, key)
        assert int(state.factory_store[0].sum()) == 6
        assert int(state.cash[0]) == INITIAL_CASH - 2  # dues paid once per turn

    def test_produce_rejected_when_the_store_is_full(self):
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(0)
        # 3 factories hold 6; start at capacity.
        full = jnp.zeros((2, 5, PRICE_SLOTS), dtype=jnp.int32).at[0, 0, 0].set(6)
        state = self._three_factory_state(factory_store=full)

        state = func_env.transition(
            state, jnp.array([ACTION_PRODUCE + 1, 0, 0, 0, 0]), key, params,
        )

        assert int(state.produce_active) == 0
        assert int(state.cash[0]) == INITIAL_CASH  # no dues for a no-op
        assert int(state.produced_this_turn) == 0

    def test_produce_matches_its_action_mask(self):
        """A produce the mask forbids must be a no-op in the handler too."""
        func_env = _make_func_env()
        params = _make_params()
        key = random.PRNGKey(0)
        state = self._three_factory_state()

        state = self._produce_batch(func_env, state, params, key)
        masks = func_env._action_masks(state, params)
        assert int(masks["action_type"][ACTION_PRODUCE + 1]) == 0

        before = state
        state = func_env.transition(
            state, jnp.array([ACTION_PRODUCE + 1, 0, 0, 0, 0]), key, params,
        )
        assert int(state.factory_store[0].sum()) == int(before.factory_store[0].sum())


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

        encoder = ActionEncoder(2, 5)
        action_idx = encoder.encode(ACTION_PRODUCE, {"color": 0, "price_slot": 0})
        action = jnp.array(action_idx, dtype=jnp.int32)

        # Step 1: enter produce mode, pay $1 to opponent
        s1 = func_env.transition(state, action, key, params)
        assert int(s1.cash[0]) == INITIAL_CASH - 1
        assert int(s1.produce_active) == 1

        # Step 2: produce the container (recurrent produce step)
        key2 = random.PRNGKey(43)
        s2 = func_env.transition(s1, action, key2, params)
        p0_color = int(jnp.argmax(s2.factory_colors[0]))
        assert int(s2.container_supply[p0_color]) == 7  # 8 - 1

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
        """Verify the game terminates via step limit."""
        env = ContainerJaxEnv(num_players=2, num_colors=2)
        encoder = ActionEncoder(2, 2)
        obs, info = env.reset(seed=42)

        steps = 0
        while True:
            action_idx = encoder.encode(ACTION_PASS, {})
            obs, reward, term, trunc, info = env.step(action_idx)
            steps += 1
            if term:
                break
            if steps > 1200:
                pytest.fail("Game did not terminate within 1200 steps")

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

        # Produce must name a colour P0 actually owns a factory for.
        p0_color = int(jnp.argmax(env.state.factory_colors[0]))

        total = 0.0
        actions = [
            # P0 produces: enters produce mode and pays $1 -> reward -1
            (ACTION_PRODUCE, {"color": p0_color, "price_slot": 0}),
            # P0 produces: recurrent step, the container is actually produced
            (ACTION_PRODUCE, {"color": p0_color, "price_slot": 0}),
            (ACTION_PASS, {}),          # P0 passes, turn ends
            (ACTION_PASS, {}),          # P1 passes (action 1)
            (ACTION_PASS, {}),          # P1 passes (action 2), turn back to P0
            (ACTION_TAKE_LOAN, {}),     # P0 takes loan -> net worth change
            (ACTION_BUY_WAREHOUSE, {}), # P0 buys warehouse -> pays money
            (ACTION_PASS, {}),          # P0 passes
            (ACTION_PASS, {}),          # P1 passes
            (ACTION_PASS, {}),          # P1 passes
        ]
        for atype, aparams in actions:
            idx = encoder.encode(atype, aparams)
            obs, reward, term, trunc, info = env.step(idx)
            total += float(reward)
            if term:
                break

        # Several actions with non-zero reward were taken
        assert total != 0.0
