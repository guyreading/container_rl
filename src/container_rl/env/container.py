"""This module provides a Container functional environment and Gymnasium environment wrapper ContainerJaxEnv."""

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


# Constants
MAX_PLAYERS = 4
MAX_CONTAINER_COLORS = 5  # typical: yellow, red, blue, green, black?
MAX_FACTORIES_PER_PLAYER = 5
MAX_WAREHOUSES_PER_PLAYER = 5
MAX_CONTAINERS_PER_COLOR = 20  # 4 * num_players
SHIP_CAPACITY = 5
FACTORY_STORE_SLOTS = 10  # price slots $1-$10
HARBOUR_STORE_SLOTS = 10
INITIAL_CASH = 20
LOAN_AMOUNT = 10
LOAN_INTEREST = 1


class EnvState(NamedTuple):
    """A named tuple which contains the full state of the Container game."""
    # Player-specific state (for the agent)
    cash: jax.Array  # scalar
    loans: jax.Array  # number of loans (0-2)
    factory_colors: jax.Array  # bitmask of owned factory colors
    warehouse_count: jax.Array  # number of warehouses
    factory_store: jax.Array  # shape (MAX_CONTAINER_COLORS, FACTORY_STORE_SLOTS) maybe
    harbour_store: jax.Array  # shape (MAX_CONTAINER_COLORS, HARBOUR_STORE_SLOTS)
    ship_contents: jax.Array  # shape (SHIP_CAPACITY,) each entry: color (0 = empty)
    ship_location: jax.Array  # 0=open sea, 1=auction island, 2+=player index (for harbour)
    island_store: jax.Array  # shape (MAX_CONTAINER_COLORS,) counts
    # Opponent state (simplified: we can have a single opponent for now)
    opponent_cash: jax.Array
    opponent_factory_colors: jax.Array
    opponent_warehouse_count: jax.Array
    opponent_factory_store: jax.Array
    opponent_harbour_store: jax.Array
    opponent_ship_contents: jax.Array
    opponent_ship_location: jax.Array
    opponent_island_store: jax.Array
    # Global state
    container_supply: jax.Array  # shape (MAX_CONTAINER_COLORS,) counts
    turn_phase: jax.Array  # 0=start of turn, 1=after first action, 2=end of turn
    current_player: jax.Array  # 0=agent, 1=opponent
    game_over: jax.Array  # 0/1
    # Secret value card (for agent)
    secret_value_color: jax.Array  # color that scores 10/5
    # For auction
    auction_active: jax.Array  # 0/1
    auction_bids: jax.Array  # placeholder
    # Misc
    step_count: jax.Array


def cmp(a, b):
    """Returns 1 if a > b, otherwise returns -1."""
    return (a > b).astype(int) - (a < b).astype(int)


@struct.dataclass
class ContainerParams:
    """Parameters for the jax Container environment."""
    num_players: int = 2
    num_colors: int = 5
    natural: bool = False  # not used, kept for compatibility


class ContainerFunctional(
    FuncEnv[EnvState, jax.Array, int, float, bool, RenderStateType, ContainerParams]
):
    """Container is a board game about shipping logistics and market speculation.

    ### Description
    Players own factories that produce colored containers, which they can sell to other
    players' harbour stores, ship to auction island, and eventually deliver to their
    island store for scoring. The game ends when the second color of containers runs out.

    This is a simplified single‑agent version where the agent plays against one
    fixed opponent (random strategy). The agent controls one player; the opponent
    follows a hard‑coded policy.

    ### Action Space
    Actions are discrete codes representing the possible moves:
    0: Buy a factory (of cheapest available color)
    1: Buy a warehouse
    2: Produce containers (and optionally reprice)
    3: Buy containers from opponent's factory store (cheapest available)
    4: Move ship to opponent's harbour and load containers (random selection)
    5: Move ship to open sea
    6: Move ship to auction island and hold auction
    7: Pass

    ### Observation Space
    The observation is a flat vector containing:
    - Agent's cash, loans, warehouse count
    - Bitmask of owned factory colors
    - Factory store counts per color/price
    - Harbour store counts per color/price
    - Ship contents (encoded)
    - Ship location
    - Island store counts per color
    - Opponent's visible state (cash, factory colors, warehouse count)
    - Container supply per color
    - Turn phase, current player
    - Secret value color (one‑hot)

    ### Rewards
    Reward is the change in net worth (cash + value of island goods - loans) after
    each turn, scaled appropriately.

    ### Arguments
    ```
    gym.make('Jax-Container-v0', num_players=2, num_colors=5)
    ```
    """

    action_space = spaces.Discrete(8)

    # Observation space placeholder – will need proper bounds
    observation_space = spaces.Box(
        low=0, high=100, shape=(100,), dtype=np.float32  # TODO: define exact shape
    )

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
        "autoreseet-mode": AutoresetMode.NEXT_STEP,
    }

    def transition(
        self,
        state: EnvState,
        action: int | jax.Array,
        key: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> EnvState:
        """The container environment's state transition function."""
        # TODO: implement game logic
        # For now, just pass through and increment step count
        new_state = EnvState(
            cash=state.cash,
            loans=state.loans,
            factory_colors=state.factory_colors,
            warehouse_count=state.warehouse_count,
            factory_store=state.factory_store,
            harbour_store=state.harbour_store,
            ship_contents=state.ship_contents,
            ship_location=state.ship_location,
            island_store=state.island_store,
            opponent_cash=state.opponent_cash,
            opponent_factory_colors=state.opponent_factory_colors,
            opponent_warehouse_count=state.opponent_warehouse_count,
            opponent_factory_store=state.opponent_factory_store,
            opponent_harbour_store=state.opponent_harbour_store,
            opponent_ship_contents=state.opponent_ship_contents,
            opponent_ship_location=state.opponent_ship_location,
            opponent_island_store=state.opponent_island_store,
            container_supply=state.container_supply,
            turn_phase=state.turn_phase,
            current_player=state.current_player,
            game_over=state.game_over,
            secret_value_color=state.secret_value_color,
            auction_active=state.auction_active,
            auction_bids=state.auction_bids,
            step_count=state.step_count + 1,
        )
        return new_state

    def initial(
        self, rng: PRNGKeyType, params: ContainerParams = ContainerParams
    ) -> EnvState:
        """Container initial observation function."""
        # TODO: set up initial state according to rules
        # For now, create a minimal placeholder state
        state = EnvState(
            cash=jnp.array(INITIAL_CASH),
            loans=jnp.array(0),
            factory_colors=jnp.array([1, 0, 0, 0, 0]),  # one factory of color 0
            warehouse_count=jnp.array(1),
            factory_store=jnp.zeros((params.num_colors, FACTORY_STORE_SLOTS)),
            harbour_store=jnp.zeros((params.num_colors, HARBOUR_STORE_SLOTS)),
            ship_contents=jnp.zeros(SHIP_CAPACITY),
            ship_location=jnp.array(0),  # open sea
            island_store=jnp.zeros(params.num_colors),
            opponent_cash=jnp.array(INITIAL_CASH),
            opponent_factory_colors=jnp.array([0, 1, 0, 0, 0]),  # different color
            opponent_warehouse_count=jnp.array(1),
            opponent_factory_store=jnp.zeros((params.num_colors, FACTORY_STORE_SLOTS)),
            opponent_harbour_store=jnp.zeros((params.num_colors, HARBOUR_STORE_SLOTS)),
            opponent_ship_contents=jnp.zeros(SHIP_CAPACITY),
            opponent_ship_location=jnp.array(0),
            opponent_island_store=jnp.zeros(params.num_colors),
            container_supply=jnp.array([params.num_players * 4] * params.num_colors),
            turn_phase=jnp.array(0),
            current_player=jnp.array(0),
            game_over=jnp.array(0),
            secret_value_color=jnp.array(0),  # color 0 is secret value
            auction_active=jnp.array(0),
            auction_bids=jnp.array(0),
            step_count=jnp.array(0),
        )
        return state

    def observation(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Container observation."""
        # TODO: flatten relevant parts of state into a vector
        # For now, return zeros
        return jnp.zeros(100, dtype=np.float32)

    def terminal(
        self,
        state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Determines if a particular Container observation is terminal."""
        # Game ends when second container color is exhausted
        # For now, just based on step count
        return (state.step_count > 1000) | (state.game_over > 0)

    def reward(
        self,
        state: EnvState,
        action: ActType,
        next_state: EnvState,
        rng: PRNGKeyType,
        params: ContainerParams = ContainerParams,
    ) -> jax.Array:
        """Calculates reward from a state."""
        # TODO: compute change in net worth
        return jnp.array(0.0)

    def render_init(
        self, screen_width: int = 800, screen_height: int = 600
    ) -> RenderStateType:
        """Returns an initial render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        # placeholder
        return screen, "", 0

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
        params: ContainerParams = ContainerParams,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an image from a state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy_text]"`'
            ) from e
        screen, _, _ = render_state
        screen_width, screen_height = 800, 600
        screen.fill((7, 99, 36))
        # placeholder: draw some text
        font = pygame.font.Font(None, 36)
        text = font.render("Container RL Environment", True, (255, 255, 255))
        screen.blit(text, (screen_width // 2 - text.get_width() // 2, screen_height // 2))
        return render_state, np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_close(
        self, render_state: RenderStateType, params: ContainerParams = ContainerParams
    ) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()

    def get_default_params(self, **kwargs) -> ContainerParams:
        """Get the default params."""
        return ContainerParams(**kwargs)


class ContainerJaxEnv(FunctionalJaxEnv, EzPickle):
    """A Gymnasium Env wrapper for the functional container env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs):
        """Initializes Gym wrapper for container functional env."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = ContainerFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    """
    Temporary environment tester function.
    """
    env = HumanRendering(ContainerJaxEnv(render_mode="rgb_array"))

    obs, info = env.reset()
    print(obs, info)

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)

    exit()