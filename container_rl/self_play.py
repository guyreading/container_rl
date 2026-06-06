"""Self-play wrapper, opponent pool, and ELO tracking for Container RL."""

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
    """Update two ELO ratings after a single game."""
    expected = expected_score(elo_a, elo_b)
    score = 1.0 if a_won else 0.0
    delta = k * (score - expected)
    return elo_a + delta, elo_b - delta


def rankings_from_net_worth(
    state, num_players: int, func_env: ContainerFunctional, num_colors: int
) -> dict[int, int]:
    """Compute final rankings (0 = winner) from net worth."""
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
    """Fixed-size pool of historical model checkpoints with ELO ratings."""

    max_size: int = 20
    _entries: list[OpponentEntry] = field(default_factory=list)

    def add(self, model_path: str, elo: float = 1000.0) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append(OpponentEntry(model_path=model_path, elo=elo))

    def sample(self, n: int, current_elo: float, device: str = "cpu") -> list[MaskablePPO]:
        if len(self._entries) == 0:
            return []
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


class RandomOpponent:
    """Fallback opponent that samples uniformly from valid (unmasked) actions."""

    def predict(
        self, obs: np.ndarray, action_masks: list[np.ndarray] | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        action = np.zeros(5, dtype=np.int64)
        if action_masks is not None:
            for i, mask in enumerate(action_masks):
                valid = np.flatnonzero(np.asarray(mask))
                action[i] = np.random.choice(valid) if len(valid) > 0 else 0
        return action, None


class SelfPlayWrapper(gym.Wrapper):
    """Wraps a multi-player Container env for single-agent self-play training.

    The training agent always occupies player slot 0.  When it is the
    agent's turn this wrapper passes through.  When it is an opponent's
    turn the wrapper auto-plays using a historical model checkpoint,
    transparently handling shopping / produce / auction continuation.

    Rewards are the training agent's **cumulative net worth change**
    across the full action‑opponent‑response cycle.
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_models: dict[int, MaskablePPO | RandomOpponent],
        main_player: int = 0,
    ) -> None:
        super().__init__(env)
        self.opponent_models = opponent_models
        self.main_player = main_player
        self._agent_reward: float = 0.0
        self._fallback = RandomOpponent()

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

    def _opponent_predict(
        self, model: MaskablePPO | RandomOpponent, deterministic: bool = False,
    ) -> np.ndarray:
        obs = self._current_observation()
        masks = self._current_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        return np.atleast_1d(action)

    def _get_model(self, player_id: int) -> MaskablePPO | RandomOpponent:
        return self.opponent_models.get(player_id, self._fallback)

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

        if term:
            info["final_rankings"] = rankings_from_net_worth(
                self._state, self._np, self._func_env, self._nc,
            )
            info["agent_rank"] = info["final_rankings"].get(self.main_player, 999)

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
        model = self._get_model(player_id)
        action = self._opponent_predict(model, deterministic=False)
        self.env.step(action)
        self._handle_continuation(model)

        if int(self._state.current_player) != player_id:
            return

        action = self._opponent_predict(model, deterministic=False)
        self.env.step(action)
        self._handle_continuation(model)

    def _handle_continuation(self, model: MaskablePPO | RandomOpponent) -> None:
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
