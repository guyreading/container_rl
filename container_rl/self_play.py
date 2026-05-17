"""Self-play wrapper, opponent pool, and ELO tracking for Container RL.

Provides a SelfPlayWrapper that makes a multi-player Container
environment appear single-agent to sb3 by auto-playing opponent turns
with historical model checkpoints.  Also includes an OpponentPool
for managing snapshots and an ELO rating system.
"""

from __future__ import annotations

import os
import random as py_random
from dataclasses import dataclass, field

import gymnasium as gym
import jax
import numpy as np
from sb3_contrib import MaskablePPO

from container_rl.env.container import ContainerFunctional, ContainerParams


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that player *a* beats player *b*."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def elo_update(elo_a: float, elo_b: float, a_won: bool, k: float = 32.0) -> tuple[float, float]:
    """Update two ELO ratings after a single game.

    Returns ``(new_elo_a, new_elo_b)``.
    """
    expected = expected_score(elo_a, elo_b)
    score = 1.0 if a_won else 0.0
    delta = k * (score - expected)
    return elo_a + delta, elo_b - delta


def rankings_from_net_worth(
    state, num_players: int, func_env: ContainerFunctional, num_colors: int
) -> dict[int, int]:
    """Compute final rankings (0 = winner) from net worth.

    Returns a dict mapping ``player_id → rank`` (lower is better).
    """
    nws = {p: int(func_env._net_worth(state, p, num_colors)) for p in range(num_players)}
    sorted_players = sorted(nws, key=nws.get, reverse=True)
    return {p: rank for rank, p in enumerate(sorted_players)}


@dataclass
class OpponentEntry:
    """A single opponent snapshot in the pool."""
    model_path: str
    elo: float = 1000.0
    games_played: int = 0


@dataclass
class OpponentPool:
    """Fixed-size pool of historical model checkpoints with ELO ratings.

    New opponents are added periodically (e.g. every 50 training updates).
    When the pool is full the oldest entry is evicted.  Sampling is biased
    towards opponents with similar ELO to provide appropriate challenges.
    """

    max_size: int = 20
    _entries: list[OpponentEntry] = field(default_factory=list)

    def add(self, model_path: str, elo: float = 1000.0) -> None:
        """Add a new opponent snapshot, evicting the oldest if full."""
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append(OpponentEntry(model_path=model_path, elo=elo))

    def sample(self, n: int, current_elo: float, device: str = "cpu") -> list[MaskablePPO]:
        """Sample *n* opponent models biased towards similar ELO.

        Returns a list of loaded ``MaskablePPO`` instances.
        """
        if len(self._entries) == 0:
            raise RuntimeError("Opponent pool is empty — add at least one snapshot")

        if len(self._entries) <= n:
            chosen = list(self._entries)
        else:
            diffs = [abs(e.elo - current_elo) for e in self._entries]
            max_diff = max(diffs) or 1.0
            scores = [1.0 - (d / max_diff) + py_random.random() * 0.3 for d in diffs]
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            chosen = [self._entries[i] for i, _ in indexed[:n]]

        models: list[MaskablePPO] = []
        for entry in chosen:
            m = MaskablePPO.load(entry.model_path, device=device)
            entry.games_played += 1
            models.append(m)
        return models

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return len(self._entries) > 0


class SelfPlayWrapper(gym.Wrapper):
    """Wraps a multi-player Container env for single-agent self-play training.

    The training agent always occupies player slot 0.  When it is the
    agent's turn this wrapper passes through.  When it is an opponent's
    turn the wrapper auto-plays using a historical model checkpoint,
    transparently handling shopping / produce / auction continuation.

    Rewards are the training agent's **cumulative net worth change**
    across the full action‑opponent‑response cycle — the agent sees its
    direct reward plus any net worth change caused by opponents before
    its next turn.
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_models: dict[int, MaskablePPO],
        main_player: int = 0,
    ) -> None:
        super().__init__(env)
        self.opponent_models = opponent_models
        self.main_player = main_player
        self._agent_reward: float = 0.0

    @property
    def _func_env(self) -> ContainerFunctional:
        return self.unwrapped.func_env

    @property
    def _state(self):
        return self._func_env.state

    @property
    def _params(self) -> ContainerParams:
        return self._func_env.params

    @property
    def _nc(self) -> int:
        return int(self._params.num_colors)

    @property
    def _np(self) -> int:
        return int(self._params.num_players)

    def _agent_nw(self) -> float:
        return float(self._func_env._net_worth(self._state, self.main_player, self._nc))

    def _current_observation(self) -> np.ndarray:
        return np.asarray(
            self._func_env.observation(self._state, jax.random.PRNGKey(0), self._params),
            dtype=np.float32,
        )

    def _current_masks(self) -> list[np.ndarray]:
        masks_dict = self._func_env._action_masks(self._state, self._params)
        return [
            np.asarray(masks_dict[k], dtype=bool)
            for k in ("action_type", "opponent", "color", "price_slot", "purchase")
        ]

    def _opponent_predict(self, model: MaskablePPO, deterministic: bool = False) -> np.ndarray:
        obs = self._current_observation()
        masks = self._current_masks()
        action, _states = model.predict(obs, action_masks=masks, deterministic=deterministic)
        return np.atleast_1d(action)

    def reset(self, **kwargs):
        self._agent_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        self._advance_to_agent_turn()
        return self._current_observation(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._agent_reward += float(reward)

        state = self._state
        if bool(state.shopping_active) or bool(state.produce_active):
            return obs, self._agent_reward, term, trunc, info

        nw_before_opponents = self._agent_nw()
        self._play_opponent_turns()
        nw_after_opponents = self._agent_nw()
        self._agent_reward += nw_after_opponents - nw_before_opponents

        term = bool(self._state.game_over)
        total_reward = self._agent_reward
        self._agent_reward = 0.0
        return self._current_observation(), total_reward, term, trunc, info

    def _advance_to_agent_turn(self) -> None:
        while True:
            if bool(self._state.game_over):
                break
            cp = int(self._state.current_player)
            if cp == self.main_player:
                break
            self._play_one_opponent_turn(cp)

    def _play_opponent_turns(self) -> None:
        self._advance_to_agent_turn()

    def _play_one_opponent_turn(self, player_id: int) -> None:
        model = self.opponent_models[player_id]

        action = self._opponent_predict(model, deterministic=False)
        self.env.step(action)
        self._handle_continuation(model)

        if int(self._state.current_player) != player_id:
            return

        action = self._opponent_predict(model, deterministic=False)
        self.env.step(action)
        self._handle_continuation(model)

    def _handle_continuation(self, model: MaskablePPO) -> None:
        state = self._state

        while bool(state.shopping_active) or bool(state.produce_active):
            action = self._opponent_predict(model, deterministic=False)
            self.env.step(action)
            state = self._state

        _round = 0
        while bool(state.auction_active):
            action = self._opponent_predict(model, deterministic=False)
            self.env.step(action)
            state = self._state
            _round += 1
            if _round > 20:
                break

    def final_rankings(self) -> dict[int, int]:
        return rankings_from_net_worth(self._state, self._np, self._func_env, self._nc)

    def final_agent_nw(self) -> float:
        return self._agent_nw()
