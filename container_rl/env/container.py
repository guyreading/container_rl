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
MAX_FACTORIES_PER_PLAYER = 4
MAX_WAREHOUSES_PER_PLAYER = 5
SHIP_CAPACITY = 5
PRICE_SLOTS = 10  # $1 through $10
PRODUCE_PRICE_CHOICES = 4  # $1-$4 prices (slots 0-3)
PRODUCE_CHOICES = PRODUCE_PRICE_CHOICES + 1  # 5 choices: $1-$4 + leave idle
LEAVE_IDLE = PRODUCE_PRICE_CHOICES  # slot index for "leave this factory idle"
HARBOUR_PRICE_MIN = 1   # slot for $2
HARBOUR_PRICE_MAX = 5   # slot for $6
HARBOUR_PRICE_CHOICES = HARBOUR_PRICE_MAX - HARBOUR_PRICE_MIN + 1  # 5 prices: $2-$6
INITIAL_CASH = 20
LOAN_AMOUNT = 10
LOAN_INTEREST = 1
FACTORY_STORAGE_MULTIPLIER = 2  # storage = factories * 2
INITIAL_CONTAINER_SUPPLY = 12  # per color

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

# Multi-head action architecture (5 heads)
HEAD_ACTION_TYPE = 0
HEAD_OPPONENT = 1
HEAD_COLOR = 2
HEAD_PRICE_SLOT = 3
HEAD_PURCHASE = 4
NUM_HEADS_FIXED = 5

# No-op constant — index 0 on every head
NO_OP = 0

# Purchase head constants (30 value slots + no-op + STOP = 32)
PURCHASE_VALUES = 30
PURCHASE_STOP = PURCHASE_VALUES + 1   # 31
PURCHASE_SIZE = PURCHASE_VALUES + 2   # 32


def num_heads(num_players: int) -> int:
    """Total number of action heads (always 5)."""
    return NUM_HEADS_FIXED


def head_sizes(num_players: int, num_colors: int) -> list[int]:
    """Per-head category counts for MultiDiscrete action space.

    Each head includes a no-op element at index 0.  During parallel
    (initial action) mode no-op is masked out on every head; during
    sequential (continuation) mode irrelevant heads are forced to no-op.

    Purchase head (always 32): index 0=no-op, 1-5=harbour $2-$6,
    6-30=auction bids, 31=STOP.  No longer depends on num_colors.
    """
    return [
        NUM_ACTION_TYPES + 1,           # action_type + no-op
        num_players,                     # opponent + no-op (0=no-op, 1..np-1=opponents)
        num_colors + 1,                  # color + no-op
        PRICE_SLOTS + 1,                 # price_slot + no-op
        PURCHASE_SIZE,                   # 32: no-op + 30 values + STOP
    ]


def mask_size(num_players: int, num_colors: int) -> int:
    """Total size of action mask vector appended to observation."""
    return (
        (NUM_ACTION_TYPES + 1)
        + num_players
        + (num_colors + 1)
        + (PRICE_SLOTS + 1)
        + PURCHASE_SIZE
    )

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

    # Shopping continuation state (for actions 3 and 4)
    shopping_active: jax.Array  # 0/1 — are we mid-shopping?
    shopping_action_type: jax.Array  # which action type (3 or 4)
    shopping_target: jax.Array  # opponent player index
    shopping_harbour_price: jax.Array  # harbour price slot from initial action

    # Produce continuation state (for recurrent produce)
    produce_active: jax.Array  # 0/1 — are we mid-produce batch?
    produce_pending: jax.Array  # [MAX_COLORS], 1=owned factory color still needs processing
    produce_was_produced: jax.Array  # 0/1, saved was_produced from first produce call

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
        self.produce_actions = num_colors * PRODUCE_CHOICES
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
            rel = action_idx - self.offsets['produce']
            color = rel // PRODUCE_CHOICES
            slot = rel % PRODUCE_CHOICES
            return ACTION_PRODUCE, {'color': color, 'price_slot': slot}

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
            return self.offsets['produce'] + params['color'] * PRODUCE_CHOICES + params['price_slot']

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

    def to_multi_head(self, action_idx: int) -> "jax.Array":
        """Convert a flat action index to a multi-head action array.

        This provides backward compatibility: old flat actions can be
        converted to the new multi-head representation for use with the
        updated environment.

        For shopping actions (3, 4) this produces the step-1 opponent
        selection action; continuation purchases must use multi-head
        actions directly.
        """
        import jax.numpy as jnp

        nc = self.num_colors
        np_ = self.num_players

        action_type, params = self.decode(action_idx)
        # action_type shifted +1 for no-op; other heads default to no-op (0)
        mh = jnp.array([action_type + 1, NO_OP, NO_OP, NO_OP, PURCHASE_STOP], dtype=jnp.int32)

        if action_type == ACTION_BUY_FACTORY:
            mh = mh.at[HEAD_COLOR].set(jnp.clip(params["color"], 0, nc - 1) + 1)

        elif action_type == ACTION_PRODUCE:
            mh = mh.at[HEAD_COLOR].set(jnp.clip(params["color"], 0, nc - 1) + 1)
            mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(params["price_slot"], 0, PRODUCE_CHOICES - 1) + 1)

        elif action_type in (ACTION_BUY_FROM_FACTORY_STORE, ACTION_MOVE_LOAD):
            opp_idx = params["opponent"] - 1
            mh = mh.at[HEAD_OPPONENT].set(jnp.clip(opp_idx, 0, np_ - 2) + 1)
            # Step 1 is opponent selection only; purchase is STOP.
            # Continuation purchases use multi-head directly (_shopping_step).

        elif action_type == ACTION_DOMESTIC_SALE:
            mh = mh.at[HEAD_COLOR].set(jnp.clip(params["color"], 0, nc - 1) + 1)
            mh = mh.at[HEAD_PRICE_SLOT].set(jnp.clip(params["price_slot"], 0, PRICE_SLOTS - 1) + 1)

        return mh


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

        self.action_space = spaces.MultiDiscrete(
            head_sizes(self.params.num_players, self.params.num_colors)
        )

        _np = self.params.num_players
        _nc = self.params.num_colors
        obs_size = (
            _np * 4                    # cash, loans, warehouse_count, ship_location per player
            + _np * _nc * 2           # factory_colors, island_store
            + _np * _nc * PRICE_SLOTS * 2  # factory_store, harbour_store
            + _np * SHIP_CAPACITY      # ship_contents
            + _nc                      # container_supply
            + 4                        # turn_phase, current_player, game_over, actions_taken
            + _np                      # secret_value_color per player
            + 5                        # auction_active, auction_seller, auction_cargo_count
            + 4                        # shopping_active, shopping_action_type, shopping_target, shopping_harbour_price
            + 1 + _nc                  # produce_active + produce_pending (nc colours)
            + mask_size(_np, _nc)     # action masks
        )
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )

        self._action_offsets = self._compute_offsets_array(_np, _nc)

    def transition(
        self,
        state: EnvState,
        action: int | jax.Array,
        key: PRNGKeyType,
        params: ContainerParams | None = None,
    ) -> EnvState:
        """Game state transition implementing full Container rules.

        Accepts both legacy flat-integer actions (from ``ActionEncoder``) and
        multi-head action arrays of shape ``(5,)``.
        """
        if params is None:
            params = self.params
        action = jnp.asarray(action, dtype=jnp.int32)
        np_ = params.num_players
        nc = params.num_colors

        if action.ndim == 0 or (action.ndim == 1 and action.shape[0] == 1):
            action = self._flat_to_multihd(
                action.reshape((),) if action.ndim == 1 else action, params
            )

        def _do_shopping(s):
            return self._shopping_step(s, action, params)

        def _do_produce_shopping(s):
            return self._produce_shopping_step(s, action, params)

        def _do_normal(s):
            s = self._pay_interest(s, np_)
            atype = jnp.clip(action[HEAD_ACTION_TYPE] - 1, 0, NUM_ACTION_TYPES - 1)
            # Capture whether produce already happened BEFORE dispatching,
            # so _advance_turn knows whether to count this as an action.
            was_produced = s.produced_this_turn > 0
            s = self._dispatch_action(s, action, key, params)
            # Save was_produced for finish_producing when the batch ends.
            s = s._replace(
                produce_was_produced=jnp.where(
                    s.produce_active > 0, was_produced, s.produce_was_produced,
                ),
            )
            # Don't advance turn if still mid-shopping, mid-produce, mid-auction,
            # or if the auction failed (no turn consumed for a bad auction attempt).
            auction_failed = (atype == ACTION_MOVE_AUCTION) & (~s.auction_active.astype(jnp.bool_))
            s = jax.lax.cond(
                auction_failed | (s.shopping_active > 0) | (s.produce_active > 0) | (s.auction_active > 0),
                lambda x: x,
                lambda x: self._advance_turn(x, atype, np_, was_produced),
                s,
            )
            return s

        def _do_auction_continue(s):
            return self._auction_continue_step(s, action, key, params)

        state = jax.lax.cond(
            state.auction_active > 0,
            _do_auction_continue,
            lambda s: jax.lax.cond(
                s.produce_active > 0,
                _do_produce_shopping,
                lambda s: jax.lax.cond(
                    s.shopping_active > 0,
                    _do_shopping,
                    _do_normal,
                    s,
                ),
                s,
            ),
            state,
        )

        state = self._check_game_end(state, nc)
        state = state._replace(step_count=state.step_count + 1)
        return state

    def _flat_to_multihd(self, action_idx: jax.Array, params: ContainerParams) -> jax.Array:
        """Convert a legacy flat action index to a multi-head action array.

        For shopping actions (3, 4) this produces the step-1 opponent
        selection action; continuation purchases must use multi-head
        actions directly.
        """
        np_ = params.num_players
        nc = params.num_colors
        action_idx = jnp.clip(action_idx, 0, self.encoder.total_actions - 1)
        action_type = jnp.searchsorted(self._action_offsets, action_idx, side="right") - 1
        action_type = jnp.clip(action_type, 0, NUM_ACTION_TYPES - 1)
        rel_offset = action_idx - self._action_offsets[action_type]

        num_hds = num_heads(np_)
        mh = jnp.zeros(num_hds, dtype=jnp.int32)
        mh = mh.at[HEAD_ACTION_TYPE].set(action_type + 1)
        mh = mh.at[HEAD_PURCHASE].set(PURCHASE_STOP)

        combos = nc * PRICE_SLOTS

        # BUY_FACTORY
        color = jnp.clip(rel_offset, 0, nc - 1)
        mh = mh.at[HEAD_COLOR].set(
            jnp.where(action_type == ACTION_BUY_FACTORY, color + 1, mh[HEAD_COLOR]))

        # BUY_FROM_FACTORY_STORE: opponent selection only (step 1)
        opp_idx = rel_offset // combos
        mh = mh.at[HEAD_OPPONENT].set(
            jnp.where(action_type == ACTION_BUY_FROM_FACTORY_STORE,
                      jnp.clip(opp_idx, 0, np_ - 2) + 1, mh[HEAD_OPPONENT]))

        # MOVE_LOAD: opponent selection only (step 1)
        mh = mh.at[HEAD_OPPONENT].set(
            jnp.where(action_type == ACTION_MOVE_LOAD,
                      jnp.clip(opp_idx, 0, np_ - 2) + 1, mh[HEAD_OPPONENT]))

        # PRODUCE: colour and price_slot for the first factory (recurrent step)
        produce_color = rel_offset // PRODUCE_CHOICES
        produce_slot = rel_offset % PRODUCE_CHOICES
        mh = mh.at[HEAD_COLOR].set(
            jnp.where(action_type == ACTION_PRODUCE,
                      jnp.clip(produce_color, 0, nc - 1) + 1, mh[HEAD_COLOR]))
        mh = mh.at[HEAD_PRICE_SLOT].set(
            jnp.where(action_type == ACTION_PRODUCE,
                      jnp.clip(produce_slot, 0, PRODUCE_CHOICES - 1) + 1, mh[HEAD_PRICE_SLOT]))

        # DOMESTIC_SALE
        remainder = rel_offset % combos
        dom_color = remainder // PRICE_SLOTS
        dom_slot = remainder % PRICE_SLOTS
        mh = mh.at[HEAD_COLOR].set(
            jnp.where(action_type == ACTION_DOMESTIC_SALE,
                      jnp.clip(dom_color, 0, nc - 1) + 1, mh[HEAD_COLOR]))
        mh = mh.at[HEAD_PRICE_SLOT].set(
            jnp.where(action_type == ACTION_DOMESTIC_SALE,
                      jnp.clip(dom_slot, 0, PRICE_SLOTS - 1) + 1, mh[HEAD_PRICE_SLOT]))

        return mh

    # ========================================================================
    # Shopping helpers (recurrent purchase for actions 3 and 4)
    # ========================================================================

    def _finish_shopping(self, state: EnvState, action_type: jax.Array, num_players: int) -> EnvState:
        """Clear shopping state and advance the turn."""
        state = state._replace(
            shopping_active=jnp.array(0, dtype=jnp.int32),
            shopping_action_type=jnp.array(0, dtype=jnp.int32),
            shopping_target=jnp.array(0, dtype=jnp.int32),
            shopping_harbour_price=jnp.array(0, dtype=jnp.int32),
        )
        return self._advance_turn(state, action_type, num_players, False)

    def _do_one_purchase(self, state: EnvState, action_type: jax.Array, target: jax.Array,
                         colour: jax.Array, source_slot: jax.Array,
                         harbour_price: jax.Array = jnp.array(0, dtype=jnp.int32),
                         ) -> EnvState:
        """Execute one container purchase.

        *colour* is the container colour (0..nc-1), *source_slot* is the
        source price slot at the opponent, and *harbour_price* is the
        destination harbour price in dollars (for factory buys only;
        ignored for ship loads).

        Returns the state with cash, stores, and ship updated if the purchase
        was valid.
        """
        player = state.current_player
        nc = state.factory_colors.shape[1]
        is_factory = action_type == ACTION_BUY_FROM_FACTORY_STORE

        color = jnp.clip(colour, 0, nc - 1)
        price_slot = jnp.clip(source_slot, 0, PRICE_SLOTS - 1)
        cost = price_slot + 1

        fs_avail = state.factory_store[target, color, price_slot]
        hs_avail = state.harbour_store[target, color, price_slot]
        available = jnp.where(is_factory, fs_avail, hs_avail)

        hs_stored = self._count_store_containers(state.harbour_store, player)
        ship_stored = self._count_ship_cargo(state.ship_contents, player)
        has_space = jnp.where(
            is_factory,
            hs_stored < state.warehouse_count[player],
            ship_stored < SHIP_CAPACITY,
        )
        can_afford = state.cash[player] >= cost
        do_buy = (available > 0) & has_space & can_afford

        new_cash = state.cash.at[player].add(jnp.where(do_buy, -cost, 0))
        new_cash = new_cash.at[target].add(jnp.where(do_buy, cost, 0))
        state = state._replace(cash=new_cash)

        state = state._replace(
            factory_store=state.factory_store.at[target, color, price_slot].set(
                jnp.where(do_buy & is_factory, fs_avail - 1,
                          state.factory_store[target, color, price_slot])
            ),
            harbour_store=state.harbour_store.at[target, color, price_slot].set(
                jnp.where(do_buy & (~is_factory), hs_avail - 1,
                          state.harbour_store[target, color, price_slot])
            ),
        )

        dest_slot = jnp.clip(harbour_price - 1, 0, PRICE_SLOTS - 1)

        state = state._replace(
            harbour_store=state.harbour_store.at[player, color, dest_slot].set(
                jnp.where(do_buy & is_factory,
                          state.harbour_store[player, color, dest_slot] + 1,
                          state.harbour_store[player, color, dest_slot])
            ),
            ship_contents=state.ship_contents.at[player, jnp.clip(ship_stored, 0, SHIP_CAPACITY - 1)].set(
                jnp.where(do_buy & (~is_factory), color + 1,
                          state.ship_contents[player, jnp.clip(ship_stored, 0, SHIP_CAPACITY - 1)])
            ),
        )

        state = state._replace(
            ship_location=state.ship_location.at[player].set(
                jnp.where(do_buy & (~is_factory) & (ship_stored == 0),
                          LOCATION_HARBOUR_OFFSET + target,
                          state.ship_location[player])
            ),
        )

        return state

    def _can_continue_shopping(self, state: EnvState, action_type: jax.Array,
                               target: jax.Array, params: ContainerParams) -> jax.Array:
        """Return True if another purchase is possible for the same action."""
        player = state.current_player
        is_factory = action_type == ACTION_BUY_FROM_FACTORY_STORE

        # Space remaining
        hs_stored = self._count_store_containers(state.harbour_store, player)
        ship_stored = self._count_ship_cargo(state.ship_contents, player)
        has_space = jnp.where(
            is_factory,
            hs_stored < state.warehouse_count[player],
            ship_stored < SHIP_CAPACITY,
        )

        # Affordable stock at target
        source = jnp.where(
            is_factory, state.factory_store[target], state.harbour_store[target],
        )
        affordable = (jnp.arange(PRICE_SLOTS) + 1) <= state.cash[player]  # (10,)
        has_affordable = jnp.any((source > 0) & affordable[None, :])

        return has_space & has_affordable

    def _shopping_step(self, state: EnvState, action: jax.Array,
                       params: ContainerParams) -> EnvState:
        """Process one purchase during shopping continuation.

        Reads ``HEAD_COLOR`` (which colour to buy) and ``HEAD_PURCHASE``
        (harbour price index 1-5 for factory, or 1=buy for ship; STOP at 31).
        The environment auto-selects the cheapest available source slot for
        the chosen colour from the target opponent.
        """
        shop_type = state.shopping_action_type
        target = state.shopping_target
        np_ = params.num_players
        nc = params.num_colors

        purchase = action[HEAD_PURCHASE]
        is_stop = purchase >= PURCHASE_STOP

        colour = jnp.clip(action[HEAD_COLOR] - 1, 0, nc - 1)

        def _do_buy(s):
            is_factory = shop_type == ACTION_BUY_FROM_FACTORY_STORE
            source_store = jnp.where(is_factory, s.factory_store[target],
                                     s.harbour_store[target])
            colour_row = source_store[colour]
            has_stock = colour_row > 0
            cheapest_slot = jnp.argmin(jnp.where(has_stock, jnp.arange(PRICE_SLOTS), PRICE_SLOTS * 10))
            cheapest_slot = jnp.clip(cheapest_slot, 0, PRICE_SLOTS - 1)

            harbour_price = jnp.where(
                is_factory,
                jnp.clip(purchase + 1, 2, 6),
                jnp.array(0, dtype=jnp.int32),
            )

            s = self._do_one_purchase(
                s, shop_type, target, colour, cheapest_slot, harbour_price,
            )
            can = self._can_continue_shopping(s, shop_type, target, params)
            s = jax.lax.cond(
                can,
                lambda x: x._replace(shopping_active=jnp.array(1, dtype=jnp.int32)),
                lambda x: self._finish_shopping(x, shop_type, np_),
                s,
            )
            return s

        return jax.lax.cond(
            is_stop,
            lambda s: self._finish_shopping(s, shop_type, np_),
            _do_buy,
            state,
        )

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _compute_offsets_array(self, num_players: int, num_colors: int) -> jax.Array:
        nc = num_colors
        np_ = num_players
        counts = [
            nc,
            1,
            nc * PRODUCE_CHOICES,
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

    # ========================================================================
    # Action masks (per-head validity vectors)
    # ========================================================================

    def _center_on_player(self, state: EnvState, player: jax.Array, num_players: int) -> EnvState:
        """Rotate per-player arrays so *player* lands at index 0.

        Player-index scalars (``auction_seller``, ``shopping_target``,
        ``current_player``) are remapped so that positional slot 0 always
        refers to the acting player.

        Global fields (``container_supply``, ``game_over``, …) are passed
        through unchanged.
        """
        p = player
        shift = -p

        state = state._replace(
            # ---- per-player arrays (roll axis 0) ---------------------------
            cash=jnp.roll(state.cash, shift, axis=0),
            loans=jnp.roll(state.loans, shift, axis=0),
            factory_colors=jnp.roll(state.factory_colors, shift, axis=0),
            warehouse_count=jnp.roll(state.warehouse_count, shift, axis=0),
            factory_store=jnp.roll(state.factory_store, shift, axis=0),
            harbour_store=jnp.roll(state.harbour_store, shift, axis=0),
            island_store=jnp.roll(state.island_store, shift, axis=0),
            ship_contents=jnp.roll(state.ship_contents, shift, axis=0),
            ship_location=jnp.roll(state.ship_location, shift, axis=0),
            secret_value_color=jnp.roll(state.secret_value_color, shift, axis=0),
            auction_bids=jnp.roll(state.auction_bids, shift, axis=0),

            # ---- player-index scalars (remap) ------------------------------
            current_player=jnp.array(0, dtype=state.current_player.dtype),
            auction_seller=(
                (state.auction_seller - p) % num_players
            ).astype(state.auction_seller.dtype),
            shopping_target=jnp.where(
                state.shopping_active > 0,
                (state.shopping_target - p) % num_players,
                jnp.array(0, dtype=state.shopping_target.dtype),
            ),
        )

        return state

    # ========================================================================
    # Action masks (per-head validity vectors)
    # ========================================================================

    def _action_masks(self, state: EnvState, params: ContainerParams) -> dict[str, jax.Array]:
        """Return binary masks for each action head (1 = valid, 0 = invalid).

        All loops are over compile-time constants (np_, nc, PRICE_SLOTS) so
        they unroll correctly under jit.  No Python ``if`` depends on traced
        values — every conditional uses :func:`jnp.where`.

        Each head has a no-op element at index 0.  The masking depends on
        the current mode:

        * **Parallel mode** (new turn): no-op masked out on every head;
          all legal non-no-op values are shown.
        * **Shopping continuation**: colour + purchase active;
          all other heads forced no-op.
        * **Produce continuation**: colour + price_slot active
          (pending factories, $1-$4 + leave idle); others forced no-op.
        * **Auction**: only action_type (forced AUCTION) and purchase
          (bid amounts) are active.
        """
        player = state.current_player
        np_ = params.num_players
        nc = params.num_colors

        cash = state.cash[player]
        fc = jnp.sum(state.factory_colors[player])
        wc = state.warehouse_count[player]
        has_space_f = self._count_store_containers(state.factory_store, player) < fc * FACTORY_STORAGE_MULTIPLIER
        has_space_h = self._count_store_containers(state.harbour_store, player) < wc
        produced = state.produced_this_turn > 0
        at_sea = state.ship_location[player] == LOCATION_OPEN_SEA
        ship_cargo = self._count_ship_cargo(state.ship_contents, player)
        ship_space = ship_cargo < SHIP_CAPACITY
        loans = state.loans[player]

        is_parallel = (state.shopping_active == 0) & (state.produce_active == 0) & (state.auction_active == 0)
        is_shopping = state.shopping_active > 0
        is_produce = state.produce_active > 0
        is_auction = state.auction_active > 0

        at_size = NUM_ACTION_TYPES + 1
        opp_size = np_
        col_size = nc + 1
        slot_size = PRICE_SLOTS + 1
        pur_size = PURCHASE_SIZE

        def _noop_only(size):
            m = jnp.zeros(size, dtype=jnp.int32)
            return m.at[NO_OP].set(1)

        # ---- action_type mask -------------------------------------------------
        at_mask = jnp.zeros(at_size, dtype=jnp.int32)

        own_all = fc >= MAX_FACTORIES_PER_PLAYER
        any_color_open = jnp.any(state.factory_colors[player] == 0)
        factory_cost = self._factory_cost(fc)
        at_mask = at_mask.at[ACTION_BUY_FACTORY + 1].set(
            jnp.where((~own_all) & any_color_open & (cash >= factory_cost), 1, 0))
        at_mask = at_mask.at[ACTION_BUY_WAREHOUSE + 1].set(
            jnp.where((wc < MAX_WAREHOUSES_PER_PLAYER) & (cash >= self._warehouse_cost(wc)), 1, 0))
        supply_ok = jnp.any((state.container_supply > 0) & (state.factory_colors[player] > 0))
        at_mask = at_mask.at[ACTION_PRODUCE + 1].set(
            jnp.where((~produced) & has_space_f & supply_ok, 1, 0))

        can_buy = jnp.zeros((), dtype=jnp.int32)
        for p in range(np_):
            is_self = p == player
            affordable_slots = jnp.arange(PRICE_SLOTS) < cash
            has_stock = jnp.any(state.factory_store[p] > 0, axis=0)
            valid = jnp.any(has_stock & affordable_slots)
            can_buy = can_buy | jnp.where(is_self, 0, valid)
        at_mask = at_mask.at[ACTION_BUY_FROM_FACTORY_STORE + 1].set(
            jnp.where(has_space_h & can_buy, 1, 0))

        can_load = jnp.zeros((), dtype=jnp.int32)
        for p in range(np_):
            is_self = p == player
            affordable_slots = jnp.arange(PRICE_SLOTS) < cash
            has_stock = jnp.any(state.harbour_store[p] > 0, axis=0)
            valid = jnp.any(has_stock & affordable_slots)
            can_load = can_load | jnp.where(is_self, 0, valid)
        at_mask = at_mask.at[ACTION_MOVE_LOAD + 1].set(
            jnp.where(ship_space & can_load, 1, 0))

        in_harbour = state.ship_location[player] >= LOCATION_HARBOUR_OFFSET
        at_mask = at_mask.at[ACTION_MOVE_SEA + 1].set(in_harbour.astype(jnp.int32))
        at_mask = at_mask.at[ACTION_MOVE_AUCTION + 1].set(
            jnp.where(state.auction_active > 0, 1,
                      jnp.where(at_sea & (ship_cargo > 0), 1, 0)))
        at_mask = at_mask.at[ACTION_PASS + 1].set(1)
        at_mask = at_mask.at[ACTION_TAKE_LOAN + 1].set(jnp.where(loans < 2, 1, 0))
        at_mask = at_mask.at[ACTION_REPAY_LOAN + 1].set(
            jnp.where((loans > 0) & (cash >= LOAN_AMOUNT), 1, 0))
        if params.use_domestic_sale:
            has_any = (jnp.sum(state.factory_store[player]) + jnp.sum(state.harbour_store[player])) > 0
            at_mask = at_mask.at[ACTION_DOMESTIC_SALE + 1].set(has_any.astype(jnp.int32))

        # ---- opponent mask ----------------------------------------------------
        opp_mask = jnp.zeros(opp_size, dtype=jnp.int32)
        for i in range(np_ - 1):
            opp = (player + 1 + i) % np_
            affordable_slots = jnp.arange(PRICE_SLOTS) < cash
            fs_ok = jnp.any(jnp.any(state.factory_store[opp] > 0, axis=0) & affordable_slots)
            hs_ok = jnp.any(jnp.any(state.harbour_store[opp] > 0, axis=0) & affordable_slots)
            opp_mask = opp_mask.at[i + 1].set(jnp.where(fs_ok | hs_ok, 1, 0))

        # ---- color mask -------------------------------------------------------
        color_mask = jnp.zeros(col_size, dtype=jnp.int32)
        # Produce continuation: show pending factory colours
        color_mask = jnp.where(is_produce,
                               jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), state.produce_pending]),
                               color_mask)
        # Shopping continuation: show colours available from target opponent
        shopping_target = state.shopping_target
        color_mask = jnp.where(is_shopping,
                               jnp.concatenate([
                                   jnp.zeros(1, dtype=jnp.int32),
                                   jnp.where(state.factory_store[shopping_target].sum(axis=1) > 0, 1, 0)
                                   | jnp.where(state.harbour_store[shopping_target].sum(axis=1) > 0, 1, 0),
                               ]),
                               color_mask)
        # Parallel: buy-factory-eligible and domestic-sale-eligible
        not_owned = state.factory_colors[player] == 0  # (nc,)
        not_owned6 = jnp.concatenate([jnp.zeros(1, dtype=jnp.bool_), not_owned])  # (nc+1,)
        color_mask = jnp.where(is_produce | is_shopping,
                               color_mask,
                               jnp.where((~own_all) & not_owned6 & (cash >= factory_cost),
                                         jnp.ones((), dtype=jnp.int32), color_mask))
        if params.use_domestic_sale:
            has_color = (jnp.sum(state.factory_store[player], axis=1)
                         + jnp.sum(state.harbour_store[player], axis=1)) > 0  # (nc,)
            has_color6 = jnp.concatenate([jnp.zeros(1, dtype=jnp.bool_), has_color])  # (nc+1,)
            color_mask = jnp.where(is_produce | is_shopping,
                                   color_mask,
                                   jnp.where(has_color6, jnp.ones((), dtype=jnp.int32), color_mask))
        # Shift to indices 1..nc
        color_mask = jnp.where(is_produce | is_shopping,
                               color_mask,
                               jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), color_mask[1:]]))

        # ---- price_slot mask --------------------------------------------------
        slot_mask = jnp.zeros(slot_size, dtype=jnp.int32)
        # Produce: $1-$4 + leave idle
        slot_mask = jnp.where(is_produce,
                              jnp.where((jnp.arange(slot_size) >= 1) & (jnp.arange(slot_size) <= PRODUCE_CHOICES), 1, 0),
                              slot_mask)
        if params.use_domestic_sale:
            for c in range(nc):
                slot_mask = jnp.where(is_produce, slot_mask,
                                      jnp.where(state.factory_store[player, c] > 0,
                                                jnp.ones(slot_size, dtype=jnp.int32), slot_mask))
                slot_mask = jnp.where(is_produce, slot_mask,
                                      jnp.where(state.harbour_store[player, c] > 0,
                                                jnp.ones(slot_size, dtype=jnp.int32), slot_mask))
        slot_mask = jnp.where(is_produce, slot_mask,
                              jnp.concatenate([jnp.zeros(1, dtype=jnp.int32),
                                               (slot_mask[1:] > 0).astype(jnp.int32)]))

        # ---- purchase mask ----------------------------------------------------
        is_factory_shop = is_shopping & (state.shopping_action_type == ACTION_BUY_FROM_FACTORY_STORE)
        is_ship_shop = is_shopping & (state.shopping_action_type == ACTION_MOVE_LOAD)

        # Base: all value slots 1..30 and STOP (31) are valid for purchase head.
        # The actual meaning depends on mode.
        pur_mask = jnp.ones(pur_size, dtype=jnp.int32)

        # During factory shopping: harbour prices $2-$6 (indices 1-5) + STOP (31)
        pur_factory_shop = jnp.zeros(pur_size, dtype=jnp.int32)
        pur_factory_shop = pur_factory_shop.at[1].set(1)   # $2
        pur_factory_shop = pur_factory_shop.at[2].set(1)   # $3
        pur_factory_shop = pur_factory_shop.at[3].set(1)   # $4
        pur_factory_shop = pur_factory_shop.at[4].set(1)   # $5
        pur_factory_shop = pur_factory_shop.at[5].set(1)   # $6
        pur_factory_shop = pur_factory_shop.at[PURCHASE_STOP].set(1)
        pur_mask = jnp.where(is_factory_shop, pur_factory_shop, pur_mask)

        # During ship shopping: only 1 (buy signal) + STOP (31)
        pur_ship_shop = jnp.zeros(pur_size, dtype=jnp.int32)
        pur_ship_shop = pur_ship_shop.at[1].set(1)
        pur_ship_shop = pur_ship_shop.at[PURCHASE_STOP].set(1)
        pur_mask = jnp.where(is_ship_shop, pur_ship_shop, pur_mask)

        # Auction: indices 0..cash = valid bids, index 1 = accept, STOP masked
        pur_auction_bid = jnp.zeros(pur_size, dtype=jnp.int32)
        pur_auction_bid = pur_auction_bid.at[NO_OP].set(1)
        pur_auction_bid = jnp.where(
            (jnp.arange(pur_size) >= 1) & (jnp.arange(pur_size) <= cash),
            jnp.ones((), dtype=jnp.int32), pur_auction_bid)
        pur_auction_bid = pur_auction_bid.at[PURCHASE_STOP].set(0)
        seller = state.auction_seller
        is_seller = is_auction & (player == seller)
        pur_seller_dec = jnp.zeros(pur_size, dtype=jnp.int32)
        pur_seller_dec = pur_seller_dec.at[NO_OP].set(1)  # reject
        pur_seller_dec = pur_seller_dec.at[1].set(1)       # accept
        pur_mask = jnp.where(is_auction,
                             jnp.where(is_seller, pur_seller_dec, pur_auction_bid),
                             pur_mask)

        # ---- Mode-based masking overrides -------------------------------------

        # Parallel mode: no-op masked out on all heads
        at_mask = jnp.where(is_parallel, at_mask.at[NO_OP].set(0), at_mask)
        opp_mask = jnp.where(is_parallel, opp_mask.at[NO_OP].set(0), opp_mask)
        color_mask = jnp.where(is_parallel, color_mask.at[NO_OP].set(0), color_mask)
        slot_mask = jnp.where(is_parallel, slot_mask.at[NO_OP].set(0), slot_mask)
        pur_mask = jnp.where(is_parallel, pur_mask.at[NO_OP].set(0), pur_mask)

        # Shopping mode: colour + purchase active, others forced no-op
        at_mask = jnp.where(is_shopping, _noop_only(at_size), at_mask)
        opp_mask = jnp.where(is_shopping, _noop_only(opp_size), opp_mask)
        color_mask = jnp.where(is_shopping, color_mask.at[NO_OP].set(0), color_mask)
        slot_mask = jnp.where(is_shopping, _noop_only(slot_size), slot_mask)
        pur_mask = jnp.where(is_shopping, pur_mask.at[NO_OP].set(0), pur_mask)

        # Produce mode: colour + price_slot active, others forced no-op
        at_mask = jnp.where(is_produce, _noop_only(at_size), at_mask)
        opp_mask = jnp.where(is_produce, _noop_only(opp_size), opp_mask)
        color_mask = jnp.where(is_produce, color_mask.at[NO_OP].set(0), color_mask)
        slot_mask = jnp.where(is_produce, slot_mask.at[NO_OP].set(0), slot_mask)
        pur_mask = jnp.where(is_produce, _noop_only(pur_size), pur_mask)

        # Auction mode: action_type + opponent (bidder) + purchase active
        at_mask = jnp.where(is_auction,
                            jnp.zeros(at_size, dtype=jnp.int32).at[ACTION_MOVE_AUCTION + 1].set(1), at_mask)
        # Opponent head is repurposed as direct player index during auction.
        # All player indices (0..np-1) must be selectable.
        opp_auction = jnp.zeros(opp_size, dtype=jnp.int32)
        for i in range(np_):
            opp_auction = opp_auction.at[i].set(1)
        opp_mask = jnp.where(is_auction, opp_auction, opp_mask)
        color_mask = jnp.where(is_auction, _noop_only(col_size), color_mask)
        slot_mask = jnp.where(is_auction, _noop_only(slot_size), slot_mask)

        return {
            "action_type": at_mask,
            "opponent": opp_mask,
            "color": color_mask,
            "price_slot": slot_mask,
            "purchase": pur_mask,
        }

    def _decode_action(self, action, params):
        action_type = jnp.searchsorted(self._action_offsets, action, side="right") - 1
        action_type = jnp.clip(action_type, 0, NUM_ACTION_TYPES - 1)
        rel_offset = action - self._action_offsets[action_type]
        return action_type, rel_offset

    def _get_target_player(self, opp_idx, current_player, num_players):
        """Map opponent relative index to actual player, clockwise.

        opp_idx=0 is the player immediately to the right, opp_idx=1 is
        two players clockwise, etc.  This matches the opponent mask
        ordering in ``_action_masks``.
        """
        return (current_player + opp_idx + 1) % num_players

    def _factory_cost(self, num_factories_owned):
        return (num_factories_owned + 1) * 3

    def _warehouse_cost(self, num_warehouses_owned):
        return num_warehouses_owned + 3

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

    def _dispatch_action(self, state, action, key, params):
        action_type = jnp.clip(action[HEAD_ACTION_TYPE] - 1, 0, NUM_ACTION_TYPES - 1)
        branches = [
            lambda s, a, k: self._action_buy_factory(s, a),
            lambda s, a, k: self._action_buy_warehouse(s, a),
            lambda s, a, k: self._action_produce(s, a, params),
            lambda s, a, k: self._action_buy_from_factory_store(s, a, params),
            lambda s, a, k: self._action_move_load(s, a, params),
            lambda s, a, k: self._action_move_sea(s, a),
            lambda s, a, k: self._action_move_auction(s, a, k, params),
            lambda s, a, k: self._action_pass(s, a),
            lambda s, a, k: self._action_take_loan(s, a),
            lambda s, a, k: self._action_repay_loan(s, a),
            lambda s, a, k: self._action_domestic_sale(s, a, params),
        ]
        return jax.lax.switch(action_type, branches, state, action, key)

    # ========================================================================
    # Action (1): Buy Factory
    # ========================================================================

    def _action_buy_factory(self, state, action):
        player = state.current_player
        nc = state.factory_colors.shape[1]
        color = jnp.clip(action[HEAD_COLOR] - 1, 0, nc - 1)

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

    def _action_buy_warehouse(self, state, action):
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

    def _action_produce(self, state, action, params):
        """Enter produce mode.  On the first call each turn this initialises
        ``produce_pending`` from the player's owned factory colours and enters
        recurrent mode.  The actual per-factory processing (colour + price)
        happens in ``_produce_shopping_step``.
        """
        player = state.current_player
        np_ = params.num_players
        nc = params.num_colors
        right_player = (player + 1) % np_

        already_produced = state.produced_this_turn > 0
        factory_count = jnp.sum(state.factory_colors[player])
        capacity = factory_count * FACTORY_STORAGE_MULTIPLIER
        current_stored = self._count_store_containers(state.factory_store, player)
        has_space = current_stored < capacity
        owned = state.factory_colors[player] > 0
        supply_ok_colors = (state.container_supply > 0) & owned
        any_possible = has_space & jnp.any(supply_ok_colors)

        do_pay = any_possible & (~already_produced)

        new_cash = state.cash.at[player].set(
            jnp.where(do_pay, state.cash[player] - 1, state.cash[player])
        )
        new_cash = new_cash.at[right_player].set(
            jnp.where(do_pay, state.cash[right_player] + 1, state.cash[right_player])
        )
        state = state._replace(cash=new_cash)

        pending = owned.astype(jnp.int32)

        state = state._replace(
            produce_active=jnp.array(1, dtype=jnp.int32),
            produce_pending=pending,
            produced_this_turn=jnp.where(
                do_pay,
                jnp.array(1, dtype=state.produced_this_turn.dtype),
                state.produced_this_turn,
            ),
        )
        return state

    # ========================================================================
    # Recurrent produce (shopping-style continuation)
    # ========================================================================

    def _produce_shopping_step(self, state, action, params):
        """Process one factory: read colour + price_slot, produce container
        or leave idle.  If more factories remain pending, stay in
        produce_active; otherwise finish and advance turn."""
        np_ = params.num_players
        nc = params.num_colors
        player = state.current_player

        color = jnp.clip(action[HEAD_COLOR] - 1, 0, nc - 1)
        slot = jnp.clip(action[HEAD_PRICE_SLOT] - 1, 0, PRODUCE_CHOICES - 1)
        is_idle = slot >= LEAVE_IDLE
        place_slot = jnp.clip(slot, 0, PRICE_SLOTS - 1)

        is_pending = (state.produce_pending[color] > 0) & (state.factory_colors[player, color] > 0)
        owns = state.factory_colors[player, color] > 0
        factory_count = jnp.sum(state.factory_colors[player])
        capacity = factory_count * FACTORY_STORAGE_MULTIPLIER
        current_stored = self._count_store_containers(state.factory_store, player)
        has_space = current_stored < capacity
        supply_ok = state.container_supply[color] > 0

        can_produce = is_pending & (~is_idle) & owns & has_space & supply_ok

        state = state._replace(
            factory_store=state.factory_store.at[player, color, place_slot].set(
                jnp.where(
                    can_produce,
                    state.factory_store[player, color, place_slot] + 1,
                    state.factory_store[player, color, place_slot],
                )
            ),
            container_supply=state.container_supply.at[color].set(
                jnp.where(
                    can_produce,
                    state.container_supply[color] - 1,
                    state.container_supply[color],
                )
            ),
        )

        # Mark this colour as processed.
        pending = state.produce_pending.at[color].set(
            jnp.where(is_pending, 0, state.produce_pending[color])
        )
        any_remaining = jnp.any(pending > 0)
        state = state._replace(produce_pending=pending)

        state = jax.lax.cond(
            any_remaining,
            lambda x: x._replace(produce_active=jnp.array(1, dtype=jnp.int32)),
            lambda x: self._finish_producing(x, np_),
            state,
        )
        return state

    def _can_continue_producing(self, state):
        return jnp.any(state.produce_pending > 0)

    def _finish_producing(self, state, num_players):
        was_produced = state.produce_was_produced > 0
        state = state._replace(
            produce_active=jnp.array(0, dtype=jnp.int32),
            produce_pending=jnp.zeros_like(state.produce_pending),
            produce_was_produced=jnp.array(0, dtype=jnp.int32),
        )
        return self._advance_turn(state, jnp.array(ACTION_PRODUCE, dtype=jnp.int32),
                                  num_players, was_produced)

    # ========================================================================
    # Action (4): Buy from Factory Store (recurrent shopping)
    # ========================================================================

    def _action_buy_from_factory_store(self, state, action, params):
        """Select an opponent to buy from (step 1 of shopping).
        Sets ``shopping_active`` if purchases are possible.
        Actual container purchases happen in ``_shopping_step``.
        """
        player = state.current_player
        np_ = params.num_players
        nc = params.num_colors

        opp_idx = jnp.clip(action[HEAD_OPPONENT] - 1, 0, np_ - 2)
        target = self._get_target_player(opp_idx, player, np_)
        target = jnp.clip(target, 0, np_ - 1)

        can = self._can_continue_shopping(
            state,
            jnp.array(ACTION_BUY_FROM_FACTORY_STORE, dtype=jnp.int32),
            target, params,
        )

        state = state._replace(
            shopping_active=jnp.where(can, jnp.array(1, dtype=jnp.int32),
                                      jnp.array(0, dtype=jnp.int32)),
            shopping_action_type=jnp.where(can, jnp.array(ACTION_BUY_FROM_FACTORY_STORE, dtype=jnp.int32),
                                           jnp.array(0, dtype=jnp.int32)),
            shopping_target=jnp.where(can, target, jnp.array(0, dtype=jnp.int32)),
        )

        return state

    # ========================================================================
    # Action (5): Move to Harbour + Load (recurrent shopping)
    # ========================================================================

    def _action_move_load(self, state, action, params):
        """Select an opponent to load from (step 1 of shopping).
        Sets ``shopping_active`` if purchases are possible.
        Actual container loads happen in ``_shopping_step``.
        """
        player = state.current_player
        np_ = params.num_players
        nc = params.num_colors

        opp_idx = jnp.clip(action[HEAD_OPPONENT] - 1, 0, np_ - 2)
        target = self._get_target_player(opp_idx, player, np_)
        target = jnp.clip(target, 0, np_ - 1)

        can = self._can_continue_shopping(
            state,
            jnp.array(ACTION_MOVE_LOAD, dtype=jnp.int32),
            target, params,
        )

        state = state._replace(
            shopping_active=jnp.where(can, jnp.array(1, dtype=jnp.int32),
                                      jnp.array(0, dtype=jnp.int32)),
            shopping_action_type=jnp.where(can, jnp.array(ACTION_MOVE_LOAD, dtype=jnp.int32),
                                           jnp.array(0, dtype=jnp.int32)),
            shopping_target=jnp.where(can, target, jnp.array(0, dtype=jnp.int32)),
        )

        return state

    # ========================================================================
    # Action (6): Move to Open Sea
    # ========================================================================

    def _action_move_sea(self, state, action):
        player = state.current_player
        state = state._replace(
            ship_location=state.ship_location.at[player].set(LOCATION_OPEN_SEA)
        )
        return state

    # ========================================================================
    # Action (7): Move to Auction Island + Hold Auction
    # ========================================================================

    def _action_move_auction(self, state, action, key, params):
        """Initiate an auction.  Snapshots cargo, clears the ship, and
        enters recurrent mode so all other players can submit bids
        concurrently."""
        player = state.current_player
        np_ = params.num_players

        at_sea = state.ship_location[player] == LOCATION_OPEN_SEA
        ship_row = state.ship_contents[player]
        has_cargo = jnp.any(ship_row > 0)
        can_auction = at_sea & has_cargo

        state = state._replace(
            auction_active=jnp.where(can_auction, jnp.array(1, dtype=jnp.int32),
                                     jnp.array(0, dtype=jnp.int32)),
            auction_seller=jnp.where(can_auction, player,
                                     jnp.zeros((), dtype=jnp.int32)),
            auction_cargo=jnp.where(can_auction, ship_row,
                                    jnp.zeros(SHIP_CAPACITY, dtype=jnp.int32)),
            auction_bids=jnp.where(
                can_auction,
                jnp.full(np_, -1, dtype=jnp.int32).at[player].set(0),
                jnp.zeros(np_, dtype=jnp.int32),
            ),
            auction_round=jnp.where(can_auction, jnp.array(0, dtype=jnp.int32),
                                    jnp.array(0, dtype=jnp.int32)),
            ship_contents=state.ship_contents.at[player].set(
                jnp.where(can_auction, jnp.zeros(SHIP_CAPACITY, dtype=jnp.int32), ship_row)
            ),
            ship_location=state.ship_location.at[player].set(
                jnp.where(can_auction, LOCATION_AUCTION_ISLAND, state.ship_location[player])
            ),
        )
        return state

    # ========================================================================
    # Recurrent auction (bidding + seller decision)
    # ========================================================================

    def _auction_continue_step(self, state, action, key, params):
        """Process one step of the recurrent auction.

        Bidding phase (auction_round == 0): any non-seller player can
        submit a blind bid via HEAD_PURCHASE at any time.  When all
        non-seller bids are collected the phase advances to the seller
        decision.

        Seller decision (auction_round == 1): the seller's HEAD_PURCHASE
        signals accept (>0) or reject (0).  The auction resolves and the
        turn ends.
        """
        np_ = params.num_players
        nc = params.num_colors
        player = state.current_player
        seller = state.auction_seller
        bidder = int(jnp.clip(action[HEAD_OPPONENT], 0, np_ - 1))
        round_ = state.auction_round
        is_bidding = round_ == 0
        is_decision = round_ == 1

        bid_amount = jnp.clip(action[HEAD_PURCHASE], 0, state.cash[int(bidder)])

        def _store_bid(s):
            # Only record if this player hasn't bid yet and is not the seller.
            has_bid = s.auction_bids[int(bidder)] >= 0
            ok_to_bid = (~has_bid) & (int(bidder) != int(seller))
            bids = s.auction_bids.at[int(bidder)].set(
                jnp.where(ok_to_bid, bid_amount, s.auction_bids[int(bidder)])
            )
            # Check if all non-seller players have bid
            all_bid = ~jnp.any(bids == -1)
            s = s._replace(auction_bids=bids)
            s = jax.lax.cond(
                all_bid,
                lambda x: x._replace(
                    auction_round=jnp.array(1, dtype=jnp.int32),
                ),
                lambda x: x,
                s,
            )
            return s

        def _resolve_auction(s):
            cargo = s.auction_cargo
            bids = s.auction_bids

            all_zero = jnp.all(bids <= 0, axis=0)
            highest_bid = jnp.max(bids)
            is_max = bids == highest_bid
            tied = jnp.sum(is_max & (bids > 0)) > 1

            winner = jnp.argmax(is_max.astype(jnp.int32))
            winner = jnp.where(all_zero, seller, winner)

            key1, key2 = random.split(key)
            tie_bids = jnp.where(
                tied,
                random.randint(key1, (np_,), 0, 6),
                jnp.zeros(np_, dtype=jnp.int32),
            )
            tie_bids = jnp.where(is_max, tie_bids, jnp.zeros((), dtype=jnp.int32) - 1)
            tie_winner = jnp.argmax(tie_bids)
            winner = jnp.where(tied, tie_winner, winner)
            effective_bid = jnp.where(tied, jnp.max(tie_bids) + highest_bid, highest_bid)

            accept = jnp.where(is_decision, bid_amount > 0, effective_bid > 0)

            winner_goods = accept & ~all_zero
            seller_goods = (~accept) | all_zero

            accept_gain = effective_bid * 2
            reject_loss = jnp.where(~accept & ~all_zero, highest_bid, 0)
            new_cash = s.cash.at[seller].add(accept_gain - reject_loss)
            winner_pays = jnp.where(winner_goods & (winner != seller), effective_bid, 0)
            new_cash = new_cash.at[winner].add(-winner_pays)

            s = s._replace(
                cash=new_cash,
                auction_active=jnp.array(0, dtype=jnp.int32),
            )

            def _deposit_goods(st, recipient, do_deposit):
                for slot in range(SHIP_CAPACITY):
                    c = cargo[slot].astype(jnp.int32)
                    color_idx = c - 1
                    valid = (c > 0).astype(jnp.int32) & do_deposit.astype(jnp.int32)
                    st = st._replace(
                        island_store=st.island_store.at[recipient, jnp.maximum(color_idx, 0)].add(
                            jnp.where(valid > 0, jnp.array(1, dtype=jnp.int32),
                                      jnp.array(0, dtype=jnp.int32))
                        )
                    )
                return st

            s = _deposit_goods(s, seller, seller_goods.astype(jnp.int32))
            s = _deposit_goods(s, winner, winner_goods.astype(jnp.int32))

            # Auction always ends the turn.
            s = self._advance_turn(s, jnp.array(ACTION_MOVE_AUCTION, dtype=jnp.int32),
                                   np_, False)
            return s

        return jax.lax.cond(
            is_bidding,
            _store_bid,
            lambda s: jax.lax.cond(
                is_decision,
                _resolve_auction,
                lambda x: x,
                s,
            ),
            state,
        )

    # ========================================================================
    # Action (8): Pass
    # ========================================================================

    def _action_pass(self, state, action):
        return state

    # ========================================================================
    # Action (9): Take Loan
    # ========================================================================

    def _action_take_loan(self, state, action):
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

    def _action_repay_loan(self, state, action):
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

    def _action_domestic_sale(self, state, action, params):
        player = state.current_player
        nc = params.num_colors

        color = jnp.clip(action[HEAD_COLOR] - 1, 0, nc - 1)
        price_slot = jnp.clip(action[HEAD_PRICE_SLOT] - 1, 0, PRICE_SLOTS - 1)

        available_factory = state.factory_store[player, color, price_slot]
        available_harbour = state.harbour_store[player, color, price_slot]

        # Try factory first, fall back to harbour (same logic as before)
        from_factory = available_factory > 0
        from_harbour_fallback = (available_factory <= 0) & (available_harbour > 0)

        do_sale = from_factory | from_harbour_fallback
        use_factory = from_factory
        use_harbour = from_harbour_fallback

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

    def _advance_turn(self, state, action_type, num_players, was_produced):
        action_type = jnp.asarray(action_type, dtype=jnp.int32)
        is_auction = action_type == ACTION_MOVE_AUCTION
        is_loan = (action_type == ACTION_TAKE_LOAN) | (action_type == ACTION_REPAY_LOAN)
        is_produce = action_type == ACTION_PRODUCE
        # Produce only consumes an action the first time (not on subsequent colours)
        consumes_extra_produce = is_produce & was_produced
        consumes_action = ((~is_loan) & (~consumes_extra_produce)).astype(jnp.int32)

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
        self, rng: PRNGKeyType, params: ContainerParams | None = None,
    ) -> EnvState:
        """Initial game state."""
        if params is None:
            params = self.params
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
            shopping_active=jnp.array(0, dtype=jnp.int32),
            shopping_action_type=jnp.array(0, dtype=jnp.int32),
            shopping_target=jnp.array(0, dtype=jnp.int32),
            shopping_harbour_price=jnp.array(0, dtype=jnp.int32),
            produce_active=jnp.array(0, dtype=jnp.int32),
            produce_pending=jnp.zeros(params.num_colors, dtype=jnp.int32),
            produce_was_produced=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        return state

    def observation(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams | None = None,
    ) -> jax.Array:
        """Convert state to an ego-centric observation for the acting player.

        Per-player arrays are rotated so that *current_player* lands at
        positional slot 0 — the policy always sees "my" data first.
        Action masks (computed on the original state) are appended at the
        end so that ``MaskablePPO`` can zero out invalid actions.
        """
        if params is None:
            params = self.params
        player = state.current_player
        num_players = params.num_players

        # Compute masks on the ORIGINAL state (already keyed to current_player internally).
        masks = self._action_masks(state, params)

        # Rotate state so the acting player is at index 0.
        centered = self._center_on_player(state, player, num_players)

        parts: list[jax.Array] = []
        parts.append(centered.cash.astype(jnp.float32))
        parts.append(centered.loans.astype(jnp.float32))
        parts.append(centered.warehouse_count.astype(jnp.float32))
        parts.append(centered.ship_location.astype(jnp.float32))

        parts.append(centered.factory_colors.reshape(-1).astype(jnp.float32))
        parts.append(centered.island_store.reshape(-1).astype(jnp.float32))
        parts.append(centered.factory_store.reshape(-1).astype(jnp.float32))
        parts.append(centered.harbour_store.reshape(-1).astype(jnp.float32))
        parts.append(centered.ship_contents.reshape(-1).astype(jnp.float32))

        parts.append(centered.container_supply.astype(jnp.float32))
        parts.append(
            jnp.array(
                [
                    centered.turn_phase.astype(jnp.float32),
                    centered.current_player.astype(jnp.float32),
                    centered.game_over.astype(jnp.float32),
                    centered.actions_taken.astype(jnp.float32),
                ]
            )
        )
        parts.append(centered.secret_value_color.astype(jnp.float32))
        parts.append(
            jnp.array(
                [
                    centered.auction_active.astype(jnp.float32),
                    centered.auction_seller.astype(jnp.float32),
                ]
            )
        )
        parts.append(jnp.sum(centered.auction_cargo > 0).astype(jnp.float32)[None])

        # Shopping continuation state (4 scalars)
        parts.append(
            jnp.array([
                centered.shopping_active.astype(jnp.float32),
                centered.shopping_action_type.astype(jnp.float32),
                centered.shopping_target.astype(jnp.float32),
                centered.shopping_harbour_price.astype(jnp.float32),
            ])
        )

        # Produce continuation state (1 + nc scalars)
        parts.append(centered.produce_active.astype(jnp.float32)[None])
        parts.append(centered.produce_pending.astype(jnp.float32))

        # Append action masks (computed pre-rotation, already player-relative)
        parts.append(masks["action_type"].astype(jnp.float32))
        parts.append(masks["opponent"].astype(jnp.float32))
        parts.append(masks["color"].astype(jnp.float32))
        parts.append(masks["price_slot"].astype(jnp.float32))
        parts.append(masks["purchase"].astype(jnp.float32))

        obs = jnp.concatenate(parts)
        obs_size = self.observation_space.shape[0]
        obs = jnp.pad(obs, (0, max(0, obs_size - obs.shape[0])), constant_values=0)[:obs_size]
        return obs.astype(jnp.float32)

    def terminal(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams | None = None,
    ) -> jax.Array:
        """Check if game is over."""
        if params is None:
            params = self.params
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
        params: ContainerParams | None = None,
    ) -> jax.Array:
        """Compute reward for the acting player (whoever ``current_player`` is)
        as their net worth change."""
        if params is None:
            params = self.params
        agent = state.current_player
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
        params: ContainerParams | None = None,
    ) -> tuple[RenderStateType, np.ndarray]:
        if params is None:
            params = self.params
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
        self, render_state: RenderStateType, params: ContainerParams | None = None,
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



