"""Container game as a Gymnasium functional JAX environment.

This implements the Container board game as described in container_rules.md.
The environment supports 2-4 players, with the agent controlling player 0.
"""

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnasium import spaces
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import EzPickle
from gymnasium.vector import AutoresetMode
from jax import random

if TYPE_CHECKING:
    import pygame

type PRNGKeyType = jax.Array
RenderStateType = tuple["pygame.Surface", str, int]  # noqa: F821


# ============================================================================
# Constants & Configuration
# ============================================================================

MAX_PLAYERS = 4
MAX_COLORS = 5
MAX_FACTORIES_PER_PLAYER = MAX_COLORS
MAX_WAREHOUSES_PER_PLAYER = 10
SHIP_CAPACITY = 5
PRICE_SLOTS = 10  # $1 through $10
INITIAL_CASH = 20
LOAN_AMOUNT = 10
LOAN_INTEREST = 1
FACTORY_STORAGE_MULTIPLIER = 2  # storage = factories * 2

# Action type indices
ACTION_BUY_FACTORY = 0
ACTION_BUY_WAREHOUSE = 1
ACTION_PRODUCE = 2
ACTION_BUY_FROM_FACTORY_STORE = 3
ACTION_MOVE_LOAD = 4
ACTION_MOVE_SEA = 5
ACTION_MOVE_AUCTION = 6
ACTION_PASS = 7
ACTION_TAKE_LOAN = 8
ACTION_REPAY_LOAN = 9
ACTION_DOMESTIC_SALE = 10
NUM_ACTION_TYPES = 11

# Ship location encoding
LOCATION_OPEN_SEA = 0
LOCATION_AUCTION_ISLAND = 1
LOCATION_HARBOUR_OFFSET = 2  # location = offset + player_index


# ============================================================================
# State Definition
# ============================================================================

class EnvState(NamedTuple):
    """Complete game state."""
    # Per-player arrays (shape: [MAX_PLAYERS, ...])
    cash: jax.Array  # [MAX_PLAYERS]
    loans: jax.Array  # [MAX_PLAYERS], number of loans (0-2)
    factory_colors: jax.Array  # [MAX_PLAYERS, MAX_COLORS], 1 if owns factory of that color
    warehouse_count: jax.Array  # [MAX_PLAYERS]

    # Stores: [MAX_PLAYERS, MAX_COLORS, PRICE_SLOTS]
    # factory_store[p, c, s] = number of containers of color c at price $(s+1) in factory store
    # harbour_store similarly
    factory_store: jax.Array
    harbour_store: jax.Array

    # Island store: [MAX_PLAYERS, MAX_COLORS] - containers delivered to island
    island_store: jax.Array

    # Ships
    ship_contents: jax.Array  # [MAX_PLAYERS, SHIP_CAPACITY], color index (0 = empty)
    ship_location: jax.Array  # [MAX_PLAYERS], encoded location

    # Global state
    container_supply: jax.Array  # [MAX_COLORS]
    turn_phase: jax.Array  # 0=start of turn (pay interest), 1=after first action, 2=end of turn
    current_player: jax.Array  # player index whose turn it is
    game_over: jax.Array  # 0/1

    # Secret value cards: [MAX_PLAYERS], color that scores 10/5 for that player
    secret_value_color: jax.Array

    # Auction state (when auction_active > 0)
    auction_active: jax.Array  # 0/1
    auction_seller: jax.Array  # player index
    auction_cargo: jax.Array  # [SHIP_CAPACITY], color index (0 = empty)
    auction_bids: jax.Array  # [MAX_PLAYERS], bid amounts
    auction_round: jax.Array  # 0=first bid, 1=second bid (tie-breaker)

    # Turn tracking
    actions_taken: jax.Array  # number of actions taken this turn (0, 1, or 2)
    produced_this_turn: jax.Array  # 0/1 whether production action already taken

    # Step count for episode limits
    step_count: jax.Array


# ============================================================================
# Action Space Design
# ============================================================================

class ActionEncoder:
    """Encode/decode between discrete action indices and their meaning."""

    def __init__(self, num_players: int, num_colors: int):
        self.num_players = num_players
        self.num_colors = num_colors

        # Calculate action counts for each group
        self.buy_factory_actions = num_colors
        self.buy_warehouse_actions = 1
        self.produce_actions = 1
        self.buy_from_factory_actions = (num_players - 1) * num_colors * PRICE_SLOTS
        self.move_load_actions = (num_players - 1) * num_colors * PRICE_SLOTS
        self.move_sea_actions = 1
        self.move_auction_actions = 1
        self.pass_actions = 1
        self.take_loan_actions = 1
        self.repay_loan_actions = 1
        self.domestic_sale_actions = 2 * num_colors * PRICE_SLOTS  # 2 store types

        # Offsets for each group
        self.offsets = {}
        offset = 0
        self.offsets['buy_factory'] = offset
        offset += self.buy_factory_actions
        self.offsets['buy_warehouse'] = offset
        offset += self.buy_warehouse_actions
        self.offsets['produce'] = offset
        offset += self.produce_actions
        self.offsets['buy_from_factory'] = offset
        offset += self.buy_from_factory_actions
        self.offsets['move_load'] = offset
        offset += self.move_load_actions
        self.offsets['move_sea'] = offset
        offset += self.move_sea_actions
        self.offsets['move_auction'] = offset
        offset += self.move_auction_actions
        self.offsets['pass'] = offset
        offset += self.pass_actions
        self.offsets['take_loan'] = offset
        offset += self.take_loan_actions
        self.offsets['repay_loan'] = offset
        offset += self.repay_loan_actions
        self.offsets['domestic_sale'] = offset
        offset += self.domestic_sale_actions

        self.total_actions = offset

    def decode(self, action_idx: int):
        """Decode action index into (action_type, params)."""
        action_idx = int(action_idx)

        # Buy factory
        if action_idx < self.offsets['buy_warehouse']:
            color = action_idx - self.offsets['buy_factory']
            return ACTION_BUY_FACTORY, {'color': color}

        # Buy warehouse
        if action_idx < self.offsets['produce']:
            return ACTION_BUY_WAREHOUSE, {}

        # Produce
        if action_idx < self.offsets['buy_from_factory']:
            return ACTION_PRODUCE, {}

        # Buy from factory store
        if action_idx < self.offsets['move_load']:
            idx = action_idx - self.offsets['buy_from_factory']
            opponent = idx // (self.num_colors * PRICE_SLOTS)
            remainder = idx % (self.num_colors * PRICE_SLOTS)
            color = remainder // PRICE_SLOTS
            price_slot = remainder % PRICE_SLOTS
            return ACTION_BUY_FROM_FACTORY_STORE, {
                'opponent': opponent + 1,  # skip player 0 (agent)
                'color': color,
                'price_slot': price_slot
            }

        # Move to harbour and load
        if action_idx < self.offsets['move_sea']:
            idx = action_idx - self.offsets['move_load']
            opponent = idx // (self.num_colors * PRICE_SLOTS)
            remainder = idx % (self.num_colors * PRICE_SLOTS)
            color = remainder // PRICE_SLOTS
            price_slot = remainder % PRICE_SLOTS
            return ACTION_MOVE_LOAD, {
                'opponent': opponent + 1,
                'color': color,
                'price_slot': price_slot
            }

        # Move to open sea
        if action_idx < self.offsets['move_auction']:
            return ACTION_MOVE_SEA, {}

        # Move to auction island
        if action_idx < self.offsets['pass']:
            return ACTION_MOVE_AUCTION, {}

        # Pass
        if action_idx < self.offsets['take_loan']:
            return ACTION_PASS, {}

        # Take loan
        if action_idx < self.offsets['repay_loan']:
            return ACTION_TAKE_LOAN, {}

        # Repay loan
        if action_idx < self.offsets['domestic_sale']:
            return ACTION_REPAY_LOAN, {}

        # Domestic sale
        idx = action_idx - self.offsets['domestic_sale']
        store_type = idx // (self.num_colors * PRICE_SLOTS)  # 0=factory, 1=harbour
        remainder = idx % (self.num_colors * PRICE_SLOTS)
        color = remainder // PRICE_SLOTS
        price_slot = remainder % PRICE_SLOTS
        return ACTION_DOMESTIC_SALE, {
            'store_type': store_type,
            'color': color,
            'price_slot': price_slot
        }

    def encode(self, action_type: int, params: dict) -> int:
        """Encode action parameters to discrete index."""
        if action_type == ACTION_BUY_FACTORY:
            return self.offsets['buy_factory'] + params['color']

        elif action_type == ACTION_BUY_WAREHOUSE:
            return self.offsets['buy_warehouse']

        elif action_type == ACTION_PRODUCE:
            return self.offsets['produce']

        elif action_type == ACTION_BUY_FROM_FACTORY_STORE:
            opponent = params['opponent'] - 1  # convert to 0-based among opponents
            idx = opponent * (self.num_colors * PRICE_SLOTS)
            idx += params['color'] * PRICE_SLOTS
            idx += params['price_slot']
            return self.offsets['buy_from_factory'] + idx

        elif action_type == ACTION_MOVE_LOAD:
            opponent = params['opponent'] - 1
            idx = opponent * (self.num_colors * PRICE_SLOTS)
            idx += params['color'] * PRICE_SLOTS
            idx += params['price_slot']
            return self.offsets['move_load'] + idx

        elif action_type == ACTION_MOVE_SEA:
            return self.offsets['move_sea']

        elif action_type == ACTION_MOVE_AUCTION:
            return self.offsets['move_auction']

        elif action_type == ACTION_PASS:
            return self.offsets['pass']

        elif action_type == ACTION_TAKE_LOAN:
            return self.offsets['take_loan']

        elif action_type == ACTION_REPAY_LOAN:
            return self.offsets['repay_loan']

        elif action_type == ACTION_DOMESTIC_SALE:
            idx = params['store_type'] * (self.num_colors * PRICE_SLOTS)
            idx += params['color'] * PRICE_SLOTS
            idx += params['price_slot']
            return self.offsets['domestic_sale'] + idx

        else:
            raise ValueError(f"Unknown action type: {action_type}")


# ============================================================================
# Parameters
# ============================================================================

@struct.dataclass
class ContainerParams:
    """Parameters for the Container environment."""
    num_players: int = 2
    num_colors: int = 5
    use_domestic_sale: bool = False  # whether to include variant action


# ============================================================================
# Main Environment
# ============================================================================

class ContainerFunctional(
    FuncEnv[EnvState, jax.Array, int, float, bool, RenderStateType, ContainerParams]
):
    """Container game as a functional JAX environment."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
        "autoreseet-mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(self, **kwargs):
        self.params = ContainerParams(**kwargs)
        self.encoder = ActionEncoder(
            num_players=self.params.num_players,
            num_colors=self.params.num_colors,
        )

        self.action_space = spaces.Discrete(self.encoder.total_actions)

        obs_size = (
            self.params.num_players * 4
            + self.params.num_players * self.params.num_colors * 2
            + self.params.num_players * self.params.num_colors * PRICE_SLOTS * 2
            + self.params.num_players * SHIP_CAPACITY
            + self.params.num_colors
            + 4
            + self.params.num_players
            + 5
        )
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )

        self._action_offsets = self._compute_offsets_array(
            self.params.num_players, self.params.num_colors
        )

    def transition(
        self,
        state: EnvState,
        action: int | jax.Array,
        key: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> EnvState:
        """Game state transition implementing full Container rules."""
        action = jnp.asarray(action, dtype=jnp.int32)
        np_ = params.num_players
        nc = params.num_colors

        state = self._pay_interest(state, np_)

        action_type, rel_offset = self._decode_action(action, params)

        state = self._dispatch_action(state, action_type, rel_offset, key, np_, nc)

        state = self._advance_turn(state, action_type, np_)

        state = self._check_game_end(state, nc)

        state = state._replace(step_count=state.step_count + 1)

        return state

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _compute_offsets_array(self, num_players: int, num_colors: int) -> jax.Array:
        nc = num_colors
        np_ = num_players
        counts = [
            nc,
            1,
            1,
            (np_ - 1) * nc * PRICE_SLOTS,
            (np_ - 1) * nc * PRICE_SLOTS,
            1,
            1,
            1,
            1,
            1,
            2 * nc * PRICE_SLOTS,
        ]
        cumsum = [0]
        for c in counts:
            cumsum.append(cumsum[-1] + c)
        return jnp.array(cumsum, dtype=jnp.int32)

    def _decode_action(self, action, params):
        action_type = jnp.searchsorted(self._action_offsets, action, side="right") - 1
        action_type = jnp.clip(action_type, 0, NUM_ACTION_TYPES - 1)
        rel_offset = action - self._action_offsets[action_type]
        return action_type, rel_offset

    def _get_target_player(self, opp_idx, current_player, num_players):
        """Map opponent relative index to actual player, skipping current player."""
        return jnp.where(opp_idx < current_player, opp_idx, opp_idx + 1)

    def _factory_cost(self, num_factories_owned):
        return (num_factories_owned + 1) * 2

    def _warehouse_cost(self, num_warehouses_owned):
        return num_warehouses_owned + 1

    def _count_store_containers(self, store, player):
        return jnp.sum(store[player])

    def _count_ship_cargo(self, ship_contents, player):
        return jnp.sum(ship_contents[player] > 0)

    # ========================================================================
    # Interest
    # ========================================================================

    def _pay_interest(self, state, num_players):
        player = state.current_player
        interest = state.loans[player] * LOAN_INTEREST
        can_pay = state.cash[player] >= interest
        new_cash = jnp.where(can_pay, state.cash[player] - interest, state.cash[player])
        state = state._replace(cash=state.cash.at[player].set(new_cash))
        return state

    # ========================================================================
    # Action dispatch
    # ========================================================================

    def _dispatch_action(self, state, action_type, rel_offset, key, np_, nc):
        branches = [
            lambda s, ro, k: self._action_buy_factory(s, ro, nc),
            lambda s, ro, k: self._action_buy_warehouse(s, ro),
            lambda s, ro, k: self._action_produce(s, ro, np_, nc),
            lambda s, ro, k: self._action_buy_from_factory_store(s, ro, np_, nc),
            lambda s, ro, k: self._action_move_load(s, ro, np_, nc),
            lambda s, ro, k: self._action_move_sea(s, ro),
            lambda s, ro, k: self._action_move_auction(s, ro, k, np_, nc),
            lambda s, ro, k: self._action_pass(s, ro),
            lambda s, ro, k: self._action_take_loan(s, ro),
            lambda s, ro, k: self._action_repay_loan(s, ro),
            lambda s, ro, k: self._action_domestic_sale(s, ro, nc),
        ]
        return jax.lax.switch(action_type, branches, state, rel_offset, key)

    # ========================================================================
    # Action (1): Buy Factory
    # ========================================================================

    def _action_buy_factory(self, state, rel_offset, num_colors):
        player = state.current_player
        color = jnp.clip(rel_offset, 0, num_colors - 1)

        num_factories = jnp.sum(state.factory_colors[player])
        already_owns = state.factory_colors[player, color] > 0
        maxed_out = num_factories >= MAX_FACTORIES_PER_PLAYER
        can_buy = (~maxed_out) & (~already_owns)
        cost = self._factory_cost(num_factories)
        can_afford = state.cash[player] >= cost
        do_buy = can_buy & can_afford

        state = state._replace(
            cash=state.cash.at[player].set(
                jnp.where(do_buy, state.cash[player] - cost, state.cash[player])
            ),
            factory_colors=state.factory_colors.at[player, color].set(
                jnp.where(do_buy, 1, state.factory_colors[player, color])
            ),
        )
        return state

    # ========================================================================
    # Action (2): Buy Warehouse
    # ========================================================================

    def _action_buy_warehouse(self, state, rel_offset):
        player = state.current_player
        num_warehouses = state.warehouse_count[player]
        can_buy = num_warehouses < MAX_WAREHOUSES_PER_PLAYER
        cost = self._warehouse_cost(num_warehouses)
        can_afford = state.cash[player] >= cost
        do_buy = can_buy & can_afford

        state = state._replace(
            cash=state.cash.at[player].set(
                jnp.where(do_buy, state.cash[player] - cost, state.cash[player])
            ),
            warehouse_count=state.warehouse_count.at[player].set(
                jnp.where(do_buy, num_warehouses + 1, num_warehouses)
            ),
        )
        return state

    # ========================================================================
    # Action (3): Produce
    # ========================================================================

    def _action_produce(self, state, rel_offset, num_players, num_colors):
        player = state.current_player
        right_player = self._get_target_player(0, player, num_players)

        already_produced = state.produced_this_turn > 0

        factory_colors_player = state.factory_colors[player]
        factory_count = jnp.sum(factory_colors_player)
        capacity = factory_count * FACTORY_STORAGE_MULTIPLIER
        current_stored = self._count_store_containers(state.factory_store, player)
        has_space = current_stored < capacity

        supply_available = state.container_supply > 0

        can_produce = (~already_produced) & has_space & (jnp.any(supply_available & (factory_colors_player > 0)))

        do_pay_union = can_produce
        new_cash = state.cash.at[player].set(
            jnp.where(do_pay_union, state.cash[player] - 1, state.cash[player])
        )
        new_cash = new_cash.at[right_player].set(
            jnp.where(do_pay_union, state.cash[right_player] + 1, state.cash[right_player])
        )
        state = state._replace(cash=new_cash)

        def _produce_one_color(i, carry):
            s, stored_so_far = carry
            owns = factory_colors_player[i] > 0
            supply_ok = state.container_supply[i] > 0
            space_ok = stored_so_far < capacity
            do_produce = can_produce & owns & supply_ok & space_ok

            place_slot = 4
            place_slot = jnp.clip(place_slot, 0, PRICE_SLOTS - 1)

            s = s._replace(
                factory_store=s.factory_store.at[player, i, place_slot].set(
                    jnp.where(
                        do_produce,
                        s.factory_store[player, i, place_slot] + 1,
                        s.factory_store[player, i, place_slot],
                    )
                ),
                container_supply=s.container_supply.at[i].set(
                    jnp.where(
                        do_produce,
                        s.container_supply[i] - 1,
                        s.container_supply[i],
                    )
                ),
            )
            stored_so_far = jnp.where(do_produce, stored_so_far + 1, stored_so_far)
            return s, stored_so_far

        stored = current_stored
        for c in range(num_colors):
            state, stored = _produce_one_color(c, (state, stored))

        state = state._replace(
            produced_this_turn=jnp.where(
                can_produce, jnp.array(1, dtype=state.produced_this_turn.dtype), state.produced_this_turn
            )
        )

        return state

    # ========================================================================
    # Action (4): Buy from Factory Store
    # ========================================================================

    def _action_buy_from_factory_store(self, state, rel_offset, num_players, num_colors):
        player = state.current_player

        opp_idx = rel_offset // (num_colors * PRICE_SLOTS)
        remainder = rel_offset % (num_colors * PRICE_SLOTS)
        color = remainder // PRICE_SLOTS
        price_slot = remainder % PRICE_SLOTS

        target = self._get_target_player(jnp.clip(opp_idx, 0, num_players - 2), player, num_players)
        target = jnp.clip(target, 0, num_players - 1)
        color = jnp.clip(color, 0, num_colors - 1)
        price_slot = jnp.clip(price_slot, 0, PRICE_SLOTS - 1)

        available = state.factory_store[target, color, price_slot]
        warehouse_limit = state.warehouse_count[player]
        current_stored = self._count_store_containers(state.harbour_store, player)
        has_space = current_stored < warehouse_limit
        can_afford = state.cash[player] >= (price_slot + 1)
        valid_target = target != player
        do_buy = (available > 0) & has_space & can_afford & valid_target

        cost = price_slot + 1

        new_cash = state.cash.at[player].add(jnp.where(do_buy, -cost, jnp.zeros((), dtype=jnp.int32)))
        new_cash = new_cash.at[target].add(jnp.where(do_buy, cost, jnp.zeros((), dtype=jnp.int32)))

        state = state._replace(
            cash=new_cash,
            factory_store=state.factory_store.at[target, color, price_slot].set(
                jnp.where(do_buy, available - 1, available)
            ),
            harbour_store=state.harbour_store.at[player, color, price_slot].set(
                jnp.where(
                    do_buy,
                    state.harbour_store[player, color, price_slot] + 1,
                    state.harbour_store[player, color, price_slot],
                )
            ),
        )
        return state

    # ========================================================================
    # Action (5): Move to Harbour + Load
    # ========================================================================

    def _action_move_load(self, state, rel_offset, num_players, num_colors):
        player = state.current_player

        opp_idx = rel_offset // (num_colors * PRICE_SLOTS)
        remainder = rel_offset % (num_colors * PRICE_SLOTS)
        color = remainder // PRICE_SLOTS
        price_slot = remainder % PRICE_SLOTS

        target = self._get_target_player(jnp.clip(opp_idx, 0, num_players - 2), player, num_players)
        target = jnp.clip(target, 0, num_players - 1)
        color = jnp.clip(color, 0, num_colors - 1)
        price_slot = jnp.clip(price_slot, 0, PRICE_SLOTS - 1)

        available = state.harbour_store[target, color, price_slot]
        ship_free = SHIP_CAPACITY - self._count_ship_cargo(state.ship_contents, player)
        has_room = ship_free > 0
        can_afford = state.cash[player] >= (price_slot + 1)
        valid_target = target != player
        do_load = (available > 0) & has_room & can_afford & valid_target

        cost = price_slot + 1

        new_loc = LOCATION_HARBOUR_OFFSET + target

        ship_row = state.ship_contents[player]
        empty_mask = ship_row <= 0
        first_empty = jnp.argmax(empty_mask.astype(jnp.int32))

        new_cash = state.cash.at[player].add(jnp.where(do_load, -cost, jnp.zeros((), dtype=jnp.int32)))
        new_cash = new_cash.at[target].add(jnp.where(do_load, cost, jnp.zeros((), dtype=jnp.int32)))

        state = state._replace(
            cash=new_cash,
            harbour_store=state.harbour_store.at[target, color, price_slot].set(
                jnp.where(do_load, available - 1, available)
            ),
            ship_contents=state.ship_contents.at[player, first_empty].set(
                jnp.where(do_load, color, state.ship_contents[player, first_empty])
            ),
            ship_location=state.ship_location.at[player].set(
                jnp.where(do_load, new_loc, state.ship_location[player])
            ),
        )
        return state

    # ========================================================================
    # Action (6): Move to Open Sea
    # ========================================================================

    def _action_move_sea(self, state, rel_offset):
        player = state.current_player
        state = state._replace(
            ship_location=state.ship_location.at[player].set(LOCATION_OPEN_SEA)
        )
        return state

    # ========================================================================
    # Action (7): Move to Auction Island + Hold Auction
    # ========================================================================

    def _action_move_auction(self, state, rel_offset, key, num_players, num_colors):
        player = state.current_player
        at_sea = state.ship_location[player] == LOCATION_OPEN_SEA
        ship_row = state.ship_contents[player]
        has_cargo = jnp.any(ship_row > 0)

        can_auction = at_sea & has_cargo

        key, subkey = random.split(key)
        cargo = ship_row
        num_cargo = jnp.sum(cargo > 0)

        bid_max = jnp.maximum(
            1,
            (jnp.array(10, dtype=jnp.int32) * num_cargo + 1).astype(jnp.int32),
        )
        raw_bids = jnp.where(
            can_auction,
            random.randint(subkey, (num_players,), 0, bid_max),
            jnp.zeros(num_players, dtype=jnp.int32),
        )
        bids = jnp.where(
            jnp.arange(num_players) == player,
            jnp.zeros((), dtype=jnp.int32),
            raw_bids,
        )

        all_zero = jnp.all(bids == 0, axis=0)

        highest_bid = jnp.max(bids)
        is_max = bids == highest_bid
        tied = jnp.sum(is_max & (bids > 0)) > 1

        winner = jnp.argmax(is_max.astype(jnp.int32))
        winner = jnp.where(all_zero, player, winner)

        tie_key, key = random.split(key)
        tie_bids = jnp.where(
            tied,
            random.randint(tie_key, (num_players,), 0, 6),
            jnp.zeros(num_players, dtype=jnp.int32),
        )
        tie_bids = jnp.where(is_max, tie_bids, jnp.zeros((), dtype=jnp.int32) - 1)
        tie_winner = jnp.argmax(tie_bids)
        winner = jnp.where(tied, tie_winner, winner)
        effective_bid = jnp.where(tied, jnp.max(tie_bids) + highest_bid, highest_bid)

        seller_accepts = effective_bid > 0

        winner_goods = seller_accepts & ~all_zero
        seller_goods = (~seller_accepts) | all_zero

        accept_gain = jnp.where(
            can_auction & seller_accepts,
            effective_bid * 2,
            jnp.zeros((), dtype=jnp.int32),
        )
        reject_loss = jnp.where(
            can_auction & (~seller_accepts) & (~all_zero),
            highest_bid,
            jnp.zeros((), dtype=jnp.int32),
        )

        new_cash = state.cash.at[player].add(accept_gain - reject_loss)
        winner_pays = jnp.where(
            can_auction & winner_goods & (winner != player),
            effective_bid,
            jnp.zeros((), dtype=jnp.int32),
        )
        new_cash = new_cash.at[winner].add(-winner_pays)

        state = state._replace(
            cash=new_cash,
            ship_contents=state.ship_contents.at[player].set(
                jnp.where(can_auction, jnp.zeros(SHIP_CAPACITY, dtype=jnp.int32), ship_row)
            ),
            ship_location=state.ship_location.at[player].set(
                jnp.where(can_auction, LOCATION_AUCTION_ISLAND, state.ship_location[player])
            ),
        )

        def _deposit_goods(s, recipient, do_deposit):
            do_dep = do_deposit.astype(jnp.int32)
            for slot in range(SHIP_CAPACITY):
                c = cargo[slot].astype(jnp.int32)
                color_idx = c - 1
                valid = (c > 0).astype(jnp.int32) & jnp.where(can_auction, do_dep, jnp.zeros_like(do_dep))
                s = s._replace(
                    island_store=s.island_store.at[recipient, jnp.maximum(color_idx, 0)].add(
                        jnp.where(valid > 0, jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
                    )
                )
            return s

        state = _deposit_goods(state, player, seller_goods.astype(jnp.int32))
        state = _deposit_goods(state, winner, winner_goods.astype(jnp.int32))

        return state

    # ========================================================================
    # Action (8): Pass
    # ========================================================================

    def _action_pass(self, state, rel_offset):
        return state

    # ========================================================================
    # Action (9): Take Loan
    # ========================================================================

    def _action_take_loan(self, state, rel_offset):
        player = state.current_player
        can_take = state.loans[player] < 2
        state = state._replace(
            loans=state.loans.at[player].set(
                jnp.where(can_take, state.loans[player] + 1, state.loans[player])
            ),
            cash=state.cash.at[player].set(
                jnp.where(can_take, state.cash[player] + LOAN_AMOUNT, state.cash[player])
            ),
        )
        return state

    # ========================================================================
    # Action (10): Repay Loan
    # ========================================================================

    def _action_repay_loan(self, state, rel_offset):
        player = state.current_player
        has_loan = state.loans[player] > 0
        can_afford = state.cash[player] >= LOAN_AMOUNT
        do_repay = has_loan & can_afford
        state = state._replace(
            loans=state.loans.at[player].set(
                jnp.where(do_repay, state.loans[player] - 1, state.loans[player])
            ),
            cash=state.cash.at[player].set(
                jnp.where(do_repay, state.cash[player] - LOAN_AMOUNT, state.cash[player])
            ),
        )
        return state

    # ========================================================================
    # Action (11): Domestic Sale (variant)
    # ========================================================================

    def _action_domestic_sale(self, state, rel_offset, num_colors):
        player = state.current_player
        num_combos = num_colors * PRICE_SLOTS
        store_type = rel_offset // num_combos
        remainder = rel_offset % num_combos
        color = remainder // PRICE_SLOTS
        price_slot = remainder % PRICE_SLOTS

        store_type = jnp.clip(store_type, 0, 1)
        color = jnp.clip(color, 0, num_colors - 1)
        price_slot = jnp.clip(price_slot, 0, PRICE_SLOTS - 1)

        available_factory = state.factory_store[player, color, price_slot]
        available_harbour = state.harbour_store[player, color, price_slot]

        from_factory = (store_type == 0) & (available_factory > 0)
        from_harbour_fallback = (store_type == 0) & (available_factory <= 0) & (available_harbour > 0)
        from_harbour_direct = (store_type == 1) & (available_harbour > 0)

        do_sale = from_factory | from_harbour_fallback | from_harbour_direct
        use_factory = from_factory
        use_harbour = from_harbour_fallback | from_harbour_direct

        state = state._replace(
            cash=state.cash.at[player].set(
                jnp.where(do_sale, state.cash[player] + 2, state.cash[player])
            ),
            factory_store=state.factory_store.at[player, color, price_slot].set(
                jnp.where(
                    use_factory,
                    available_factory - 1,
                    available_factory,
                )
            ),
            harbour_store=state.harbour_store.at[player, color, price_slot].set(
                jnp.where(
                    use_harbour,
                    available_harbour - 1,
                    available_harbour,
                )
            ),
            container_supply=state.container_supply.at[color].set(
                jnp.where(do_sale & use_factory, state.container_supply[color] + 1, state.container_supply[color])
            ),
        )
        return state

    # ========================================================================
    # Turn advancement
    # ========================================================================

    def _advance_turn(self, state, action_type, num_players):
        is_auction = action_type == ACTION_MOVE_AUCTION
        is_loan = (action_type == ACTION_TAKE_LOAN) | (action_type == ACTION_REPAY_LOAN)
        consumes_action = (~is_loan).astype(jnp.int32)

        new_actions = state.actions_taken + consumes_action
        turn_ends = (new_actions >= 2) | is_auction

        state = state._replace(
            actions_taken=jnp.where(
                turn_ends,
                jnp.array(0, dtype=state.actions_taken.dtype),
                new_actions,
            ),
            produced_this_turn=jnp.where(
                turn_ends,
                jnp.array(0, dtype=state.produced_this_turn.dtype),
                state.produced_this_turn,
            ),
            current_player=jnp.where(
                turn_ends,
                (state.current_player + 1) % num_players,
                state.current_player,
            ),
            turn_phase=jnp.where(
                turn_ends,
                jnp.array(0, dtype=state.turn_phase.dtype),
                state.turn_phase + consumes_action,
            ),
        )
        return state

    # ========================================================================
    # Game end check
    # ========================================================================

    def _check_game_end(self, state, num_colors):
        exhausted = jnp.sum(state.container_supply <= 0)
        game_over = (exhausted >= 2) | (state.step_count > 1000)
        state = state._replace(
            game_over=jnp.where(game_over, jnp.array(1, dtype=state.game_over.dtype), state.game_over)
        )
        return state

    def initial(
        self, rng: PRNGKeyType, params: ContainerParams = ContainerParams
    ) -> EnvState:
        """Initial game state."""
        # Randomly assign secret value cards
        key1, key2 = random.split(rng)
        secret_value_color = random.randint(
            key1, (params.num_players,), 0, params.num_colors
        )

        # Randomly assign starting factory colors (unique per player)
        factory_colors = jnp.zeros((params.num_players, params.num_colors), dtype=jnp.int32)
        for i in range(params.num_players):
            color = i % params.num_colors
            factory_colors = factory_colors.at[i, color].set(1)

        # Initial factory store: each player gets 1 container of their factory color at price $2
        factory_store = jnp.zeros((params.num_players, params.num_colors, PRICE_SLOTS), dtype=jnp.int32)
        for i in range(params.num_players):
            color = i % params.num_colors
            factory_store = factory_store.at[i, color, 1].set(1)  # price slot 1 = $2

        # Container supply: 4 per player per color
        container_supply = jnp.full(params.num_colors, params.num_players * 4, dtype=jnp.int32)

        state = EnvState(
            cash=jnp.full(params.num_players, INITIAL_CASH, dtype=jnp.int32),
            loans=jnp.zeros(params.num_players, dtype=jnp.int32),
            factory_colors=factory_colors,
            warehouse_count=jnp.ones(params.num_players, dtype=jnp.int32),
            factory_store=factory_store,
            harbour_store=jnp.zeros((params.num_players, params.num_colors, PRICE_SLOTS), dtype=jnp.int32),
            island_store=jnp.zeros((params.num_players, params.num_colors), dtype=jnp.int32),
            ship_contents=jnp.zeros((params.num_players, SHIP_CAPACITY), dtype=jnp.int32),
            ship_location=jnp.zeros(params.num_players, dtype=jnp.int32),
            container_supply=container_supply,
            turn_phase=jnp.array(0, dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(0, dtype=jnp.int32),
            secret_value_color=secret_value_color,
            auction_active=jnp.array(0, dtype=jnp.int32),
            auction_seller=jnp.array(0, dtype=jnp.int32),
            auction_cargo=jnp.zeros(SHIP_CAPACITY, dtype=jnp.int32),
            auction_bids=jnp.zeros(params.num_players, dtype=jnp.int32),
            auction_round=jnp.array(0, dtype=jnp.int32),
            actions_taken=jnp.array(0, dtype=jnp.int32),
            produced_this_turn=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        return state

    def observation(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Convert state to observation vector (agent perspective, player 0)."""
        parts = []

        parts.append(state.cash.astype(jnp.float32))
        parts.append(state.loans.astype(jnp.float32))
        parts.append(state.warehouse_count.astype(jnp.float32))
        parts.append(state.ship_location.astype(jnp.float32))

        parts.append(state.factory_colors.reshape(-1).astype(jnp.float32))
        parts.append(state.island_store.reshape(-1).astype(jnp.float32))
        parts.append(state.factory_store.reshape(-1).astype(jnp.float32))
        parts.append(state.harbour_store.reshape(-1).astype(jnp.float32))
        parts.append(state.ship_contents.reshape(-1).astype(jnp.float32))

        parts.append(state.container_supply.astype(jnp.float32))
        parts.append(
            jnp.array(
                [
                    state.turn_phase.astype(jnp.float32),
                    state.current_player.astype(jnp.float32),
                    state.game_over.astype(jnp.float32),
                    state.actions_taken.astype(jnp.float32),
                ]
            )
        )
        parts.append(state.secret_value_color.astype(jnp.float32))
        parts.append(
            jnp.array(
                [
                    state.auction_active.astype(jnp.float32),
                    state.auction_seller.astype(jnp.float32),
                ]
            )
        )
        parts.append(jnp.sum(state.auction_cargo > 0).astype(jnp.float32)[None])

        obs = jnp.concatenate(parts)
        obs_size = self.observation_space.shape[0]
        obs = jnp.pad(obs, (0, max(0, obs_size - obs.shape[0])), constant_values=0)[:obs_size]
        return obs.astype(jnp.float32)

    def terminal(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Check if game is over."""
        return state.game_over > 0

    def _net_worth(self, state, player, num_colors):
        """Compute net worth for a player (JAX-compatible)."""
        cash = state.cash[player].astype(jnp.int32)
        loans_penalty = state.loans[player].astype(jnp.int32) * 11

        harbour_val = jnp.sum(state.harbour_store[player]).astype(jnp.int32) * 2
        ship_val = self._count_ship_cargo(state.ship_contents, player).astype(jnp.int32) * 3

        secret_color = state.secret_value_color[player].astype(jnp.int32)
        island_row = state.island_store[player]
        has_all_colors = jnp.all(island_row > 0)

        value_per_color = jnp.full(num_colors, 2, dtype=jnp.int32)
        value_per_color = value_per_color.at[secret_color].set(
            jnp.where(has_all_colors, 10, 5)
        )

        counts = island_row
        max_count = jnp.max(counts)
        is_max_int = (counts == max_count).astype(jnp.int32)
        num_max = jnp.sum(is_max_int)
        is_tied = (num_max > 1).astype(jnp.int32)
        is_secret_in_tie = is_tied * is_max_int[secret_color]

        discard_col = jnp.zeros((), dtype=jnp.int32)
        discard_col = jnp.where(is_tied == 0, jnp.argmax(is_max_int), discard_col)

        secret_tie = (is_secret_in_tie > 0)
        discard_col = jnp.where(secret_tie, secret_color, discard_col)

        not_secret_tie = (is_tied > 0) & (~secret_tie)
        cheap_vals = jnp.where(
            is_max_int > 0, value_per_color, jnp.array(999999, dtype=jnp.int32)
        )
        cheapest_idx = jnp.argmin(cheap_vals)
        discard_col = jnp.where(not_secret_tie, cheapest_idx, discard_col)

        has_any = jnp.any(counts > 0)
        final_discard = jnp.where(
            has_any,
            jnp.zeros(num_colors, dtype=jnp.int32).at[discard_col].set(1),
            jnp.zeros(num_colors, dtype=jnp.int32),
        )

        scored_counts = jnp.where(final_discard > 0, jnp.array(0, dtype=jnp.int32), counts)
        island_val = jnp.sum(scored_counts * value_per_color)

        total = cash + harbour_val + ship_val + island_val - loans_penalty
        return total

    def reward(
        self,
        state: EnvState,
        action: ActType,
        next_state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Compute reward for agent (player 0) as net worth change."""
        agent = 0
        curr_nw = self._net_worth(state, agent, params.num_colors)
        next_nw = self._net_worth(next_state, agent, params.num_colors)
        return (next_nw - curr_nw).astype(jnp.float32)

    # Rendering methods (placeholder)
    def render_init(
        self, screen_width: int = 800, screen_height: int = 600
    ) -> RenderStateType:
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        return screen, "", 0

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
        params: ContainerParams = ContainerParams,
    ) -> tuple[RenderStateType, np.ndarray]:
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy_text]"`'
            ) from e
        screen, _, _ = render_state
        screen_width, screen_height = 800, 600
        screen.fill((7, 99, 36))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Container Game - Player {state.current_player}'s turn", True, (255, 255, 255))
        screen.blit(text, (screen_width // 2 - text.get_width() // 2, screen_height // 2))
        return render_state, np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_close(
        self, render_state: RenderStateType, params: ContainerParams = ContainerParams
    ) -> None:
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()

    def get_default_params(self, **kwargs) -> ContainerParams:
        return ContainerParams(**kwargs)


# ============================================================================
# Gymnasium Wrapper
# ============================================================================

class ContainerJaxEnv(FunctionalJaxEnv, EzPickle):
    """Gymnasium wrapper for ContainerFunctional."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = ContainerFunctional(**kwargs)
        env.transform(jax.jit)
        super().__init__(env, metadata=self.metadata, render_mode=render_mode)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    env = ContainerJaxEnv(num_players=2, num_colors=5)
    encoder = ActionEncoder(num_players=2, num_colors=5)

    print(f"Total actions: {encoder.total_actions}")
    print(f"Action space: {env.action_space}")

    # Test encoding/decoding
    test_actions = [
        (ACTION_BUY_FACTORY, {'color': 0}),
        (ACTION_BUY_FROM_FACTORY_STORE, {'opponent': 1, 'color': 2, 'price_slot': 3}),
        (ACTION_MOVE_LOAD, {'opponent': 1, 'color': 4, 'price_slot': 9}),
        (ACTION_DOMESTIC_SALE, {'store_type': 0, 'color': 1, 'price_slot': 5}),
    ]

    for action_type, params in test_actions:
        idx = encoder.encode(action_type, params)
        decoded_type, decoded_params = encoder.decode(idx)
        print(f"Encoded {action_type} {params} -> {idx}")
        print(f"Decoded {idx} -> {decoded_type} {decoded_params}")
        assert action_type == decoded_type
        assert params == decoded_params

    print("Encoding/decoding test passed!")
