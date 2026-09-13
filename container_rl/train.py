"""PPO training script with optional self-play, action masking, and TensorBoard.

Usage:
    uv run python -m container_rl.train
    uv run python -m container_rl.train --num-players 3,4,5
    uv run python -m container_rl.train --self-play --snapshot-every 50

Observation and action shapes do not depend on the player count, so every
vectorised env cycles through the ``--num-players`` counts, one per episode
(3 -> 4 -> 5 -> 3 ... by default), and the resulting policy can play any of them.

Outside ``--self-play`` the one policy plays every seat: each step's observation
is centred on the seat whose decision it is, and its reward is that seat's
net-worth change.
"""

import argparse
import os
from functools import partial
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from gymnasium.vector import SyncVectorEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from container_rl import ContainerEnv
from container_rl.env.container import MAX_PLAYERS, head_sizes, mask_size
from container_rl.self_play import (
    OpponentPool,
    SelfPlayWrapper,
    elo_update,
)

NUM_PLAYERS = MAX_PLAYERS
# Player counts the training games rotate through, one per episode.
DEFAULT_PLAYER_COUNTS = (3, 4, 5)
NUM_COLORS = 5


class ContainerMaskWrapper(gym.ObservationWrapper):
    """Extracts per-head action masks from the observation and exposes them."""

    def __init__(self, env: gym.Env, num_players: int = NUM_PLAYERS, num_colors: int = NUM_COLORS):
        super().__init__(env)
        self._num_players = num_players
        self._num_colors = num_colors
        self._msk_size = mask_size(num_players, num_colors)
        self._head_sizes = head_sizes(num_players, num_colors)

        raw_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(raw_shape - self._msk_size,), dtype=np.float32,
        )
        self._raw_obs: Optional[np.ndarray] = None

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        self._raw_obs = obs
        return obs[: -self._msk_size]

    def action_masks(self) -> np.ndarray:
        """Return the concatenated action mask as a single flat array.

        sb3's ``env_method('action_masks')`` collects results from all
        sub-envs and stacks them with ``np.stack``, so we must return a
        single 1-d array, not a list of per-head arrays.
        """
        if self._raw_obs is None:
            return np.concatenate([np.ones(s, dtype=bool) for s in self._head_sizes])
        return self._raw_obs[-self._msk_size:].astype(bool)


class PlayerCountCycler(gym.Env):
    """Plays each new episode at the next player count in a fixed rotation.

    The observation and action spaces are the same at every player count, so
    one policy can train across table sizes.  The JAX env's internal shapes
    still depend on the count, though, so a single ``ContainerEnv`` cannot be
    resized in place: this keeps one env per count and switches on ``reset``.
    Vectorised envs auto-reset when an episode ends, so each finished game
    moves that env on to the next count.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        player_counts: Sequence[int],
        make_env: Callable[[int], gym.Env],
        start: int = 0,
    ):
        super().__init__()
        if not player_counts:
            raise ValueError("PlayerCountCycler needs at least one player count")
        self.player_counts = list(player_counts)
        self._envs = [make_env(n) for n in self.player_counts]
        first = self._envs[0]
        for n, env in zip(self.player_counts, self._envs):
            if (env.observation_space != first.observation_space
                    or env.action_space != first.action_space):
                raise ValueError(
                    f"the {n}-player env's spaces differ from the {self.player_counts[0]}-player "
                    f"env's, so one policy cannot drive both"
                )
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        # reset() advances before playing, so park one step behind ``start``.
        self._idx = (start - 1) % len(self._envs)
        # A seed only arrives with the first reset; give each count its own
        # derived seed the first time that count is played.
        self._pending_seeds: dict[int, int] = {}

    @property
    def num_players(self) -> int:
        return self.player_counts[self._idx]

    @property
    def active_env(self) -> gym.Env:
        return self._envs[self._idx]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            super().reset(seed=seed)
            self._pending_seeds = {i: seed + i for i in range(len(self._envs))}
        self._idx = (self._idx + 1) % len(self._envs)
        obs, info = self.active_env.reset(
            seed=self._pending_seeds.pop(self._idx, None), options=options,
        )
        return obs, {**info, "num_players": self.num_players}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active_env.step(action)
        return obs, reward, terminated, truncated, {**info, "num_players": self.num_players}

    def action_masks(self) -> np.ndarray:
        return self.active_env.action_masks()

    def close(self) -> None:
        for env in self._envs:
            env.close()


def make_env(
    num_players: int,
    *,
    self_play: bool = False,
    opponent_pool: Optional[OpponentPool] = None,
    device: str = "cpu",
) -> gym.Env:
    """One training env at a fixed player count, masks split out of the obs."""
    env = ContainerEnv(num_players=num_players, num_colors=NUM_COLORS)
    if self_play:
        models = opponent_pool.sample(num_players - 1, 1000.0, device=device) if opponent_pool else []
        opponent_models = {i + 1: m for i, m in enumerate(models)}
        env = SelfPlayWrapper(env, opponent_models, main_player=0)
    return ContainerMaskWrapper(env, num_players=num_players, num_colors=NUM_COLORS)


def make_cycling_env(player_counts: Sequence[int], start: int = 0, **make_env_kwargs) -> gym.Env:
    """A monitored env that plays successive episodes at successive player counts."""
    cycler = PlayerCountCycler(
        player_counts, partial(make_env, **make_env_kwargs), start=start,
    )
    return Monitor(cycler)  # logs episode rewards/steps to TensorBoard


class SelfPlayCallback(BaseCallback):
    """Snapshot the model and update ELO ratings during self-play training."""

    def __init__(
        self,
        opponent_pool: OpponentPool,
        log_path: str,
        snapshot_every: int = 50,
        elo_k: float = 32.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.snapshot_dir = os.path.join(log_path, "opponents")
        self.snapshot_every = snapshot_every
        self.elo_k = elo_k
        self.agent_elo: float = 1000.0
        self._last_snapshot_timesteps: int = 0
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        n_steps = getattr(self.model, "n_steps", 128)
        n_envs = getattr(self.training_env, "num_envs", 4)
        steps_per_update = n_steps * n_envs

        if self.model.num_timesteps - self._last_snapshot_timesteps >= self.snapshot_every * steps_per_update:
            self._last_snapshot_timesteps = self.model.num_timesteps
            self._snapshot()

        infos = self.locals.get("infos", [])
        for info in (infos if isinstance(infos, list) else []):
            if isinstance(info, dict) and "final_rankings" in info:
                rankings = info["final_rankings"]
                agent_rank = info.get("agent_rank", 999)
                agent_won = agent_rank == 0
                for entry in self.opponent_pool._entries:
                    self.agent_elo, entry.elo = elo_update(
                        self.agent_elo, entry.elo, agent_won, self.elo_k,
                    )
        return True

    def _snapshot(self) -> None:
        path = os.path.join(self.snapshot_dir, f"snapshot_{self.model.num_timesteps}")
        self.model.save(path)
        self.opponent_pool.add(path, self.agent_elo)
        if self.verbose:
            print(f"\n[SelfPlay] snapshot #{len(self.opponent_pool)} saved to {path} (ELO {self.agent_elo:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on Container environment")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--num-players", type=str, default=",".join(map(str, DEFAULT_PLAYER_COUNTS)),
        help="Comma-separated player counts each env rotates through, one per "
             "episode (default: %(default)s). Each must be 2-" + str(MAX_PLAYERS) + ".",
    )
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--run-name", type=str, default="container-ppo")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--opponent-pool-size", type=int, default=20)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--snapshot-every", type=int, default=50)

    args = parser.parse_args()
    player_counts = [int(x) for x in args.num_players.split(",") if x.strip()]
    if not player_counts or any(not 2 <= n <= MAX_PLAYERS for n in player_counts):
        parser.error(f"--num-players values must each be between 2 and {MAX_PLAYERS}")
    log_path = os.path.join(args.log_dir, args.run_name)
    opponent_pool = OpponentPool(max_size=args.opponent_pool_size)

    def _make_env(start: int) -> gym.Env:
        return make_cycling_env(
            player_counts, start=start,
            self_play=args.self_play, opponent_pool=opponent_pool, device=args.device,
        )

    # Stagger the starting count so the vectorised envs are spread across
    # table sizes at any moment rather than all switching in lockstep.
    vec_env = DummyVecEnv([partial(_make_env, i) for i in range(args.num_envs)])
    vec_env.seed(args.seed)

    # The eval env rotates too, so evaluation episodes span every count.
    eval_env = _make_env(0)
    eval_env.reset(seed=args.seed + 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=log_path,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    callbacks = [eval_callback]

    if args.self_play:
        callbacks.append(SelfPlayCallback(
            opponent_pool=opponent_pool,
            log_path=log_path,
            snapshot_every=args.snapshot_every,
            elo_k=args.elo_k,
            verbose=1,
        ))

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate=args.learning_rate, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        tensorboard_log=log_path, seed=args.seed, verbose=1, device=args.device,
    )

    print(f"\nMaskablePPO training: {args.run_name}")
    print(f"  Environments: {args.num_envs}  Total timesteps: {args.total_timesteps}")
    print(f"  Player counts (rotated per episode): {player_counts}")
    print(f"  Action space: {vec_env.action_space}")
    print(f"  Observation shape: {vec_env.observation_space.shape}")
    if args.self_play:
        print(f"  Self-play: snapshot every {args.snapshot_every} updates, "
              f"pool size {args.opponent_pool_size}, ELO k={args.elo_k}")
    print()

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(log_path, "final_model"))
    vec_env.close()
    eval_env.close()
    print(f"\nTraining complete. Model saved to {log_path}")


if __name__ == "__main__":
    main()
