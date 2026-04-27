"""Container game as a Gymnasium functional JAX environment.

This implements the Container board game as described in container_rules.md.
The environment supports 2-4 players, with the agent controlling player 0.
"""

import math
import os
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import random

from gymnasium import spaces
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import EzPickle, seeding
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers import HumanRendering

PRNGKeyType: TypeAlias = jax.Array
RenderStateType = tuple["pygame.Surface", str, int]  # type: ignore  # noqa: F821


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
            num_colors=self.params.num_colors
        )
        
        # Action space is discrete with total number of possible actions
        self.action_space = spaces.Discrete(self.encoder.total_actions)
        
        # Observation space: need to define properly
        # For now, placeholder
        obs_size = (
            self.params.num_players * 4 +  # cash, loans, warehouse_count, ship_location
            self.params.num_players * self.params.num_colors * 2 +  # factory_colors, island_store
            self.params.num_players * self.params.num_colors * PRICE_SLOTS * 2 +  # factory_store, harbour_store
            self.params.num_players * SHIP_CAPACITY +  # ship_contents
            self.params.num_colors +  # container_supply
            4 +  # turn_phase, current_player, game_over, actions_taken
            self.params.num_players +  # secret_value_color
            5  # auction state
        )
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )
    
    def transition(
        self,
        state: EnvState,
        action: int | jax.Array,
        key: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> EnvState:
        """Game state transition."""
        # TODO: implement full game logic
        # For now, just increment step count and rotate player
        new_state = EnvState(
            cash=state.cash,
            loans=state.loans,
            factory_colors=state.factory_colors,
            warehouse_count=state.warehouse_count,
            factory_store=state.factory_store,
            harbour_store=state.harbour_store,
            island_store=state.island_store,
            ship_contents=state.ship_contents,
            ship_location=state.ship_location,
            container_supply=state.container_supply,
            turn_phase=state.turn_phase,
            current_player=(state.current_player + 1) % params.num_players,
            game_over=state.game_over,
            secret_value_color=state.secret_value_color,
            auction_active=state.auction_active,
            auction_seller=state.auction_seller,
            auction_cargo=state.auction_cargo,
            auction_bids=state.auction_bids,
            auction_round=state.auction_round,
            actions_taken=state.actions_taken,
            produced_this_turn=state.produced_this_turn,
            step_count=state.step_count + 1,
        )
        return new_state
    
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
        factory_colors = jnp.zeros((params.num_players, params.num_colors))
        # Simple assignment: player i gets color i % num_colors
        for i in range(params.num_players):
            color = i % params.num_colors
            factory_colors = factory_colors.at[i, color].set(1)
        
        # Initial factory store: each player gets 1 container of their factory color at price $2
        factory_store = jnp.zeros((params.num_players, params.num_colors, PRICE_SLOTS))
        for i in range(params.num_players):
            color = i % params.num_colors
            factory_store = factory_store.at[i, color, 1].set(1)  # price slot 1 = $2
        
        # Container supply: 4 per player per color
        container_supply = jnp.full(params.num_colors, params.num_players * 4)
        
        state = EnvState(
            cash=jnp.full(params.num_players, INITIAL_CASH),
            loans=jnp.zeros(params.num_players, dtype=jnp.int32),
            factory_colors=factory_colors,
            warehouse_count=jnp.ones(params.num_players, dtype=jnp.int32),
            factory_store=factory_store,
            harbour_store=jnp.zeros((params.num_players, params.num_colors, PRICE_SLOTS)),
            island_store=jnp.zeros((params.num_players, params.num_colors)),
            ship_contents=jnp.zeros((params.num_players, SHIP_CAPACITY)),
            ship_location=jnp.zeros(params.num_players, dtype=jnp.int32),
            container_supply=container_supply,
            turn_phase=jnp.array(0),
            current_player=jnp.array(0),
            game_over=jnp.array(0),
            secret_value_color=secret_value_color,
            auction_active=jnp.array(0),
            auction_seller=jnp.array(0),
            auction_cargo=jnp.zeros(SHIP_CAPACITY),
            auction_bids=jnp.zeros(params.num_players),
            auction_round=jnp.array(0),
            actions_taken=jnp.array(0),
            produced_this_turn=jnp.array(0),
            step_count=jnp.array(0),
        )
        return state
    
    def observation(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Convert state to observation vector."""
        # TODO: implement proper observation
        return jnp.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def terminal(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Check if game is over."""
        # Game ends when second container color is exhausted
        exhausted_colors = jnp.sum(state.container_supply <= 0)
        return (exhausted_colors >= 2) | (state.step_count > 1000)
    
    def reward(
        self,
        state: EnvState,
        action: ActType,
        next_state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Compute reward for agent (player 0)."""
        # TODO: compute net worth change
        return jnp.array(0.0)
    
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