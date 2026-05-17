"""PPO training script with optional self-play, action masking, and TensorBoard.

Usage:
    uv run python -m container_rl.train
    uv run python -m container_rl.train --self-play --snapshot-every 50
"""

import argparse
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from container_rl import ContainerEnv
from container_rl.env.container import head_sizes, mask_size
from container_rl.self_play import (
    OpponentPool,
    SelfPlayWrapper,
    elo_update,
)

NUM_PLAYERS = 4
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

    def action_masks(self) -> list[np.ndarray]:
        if self._raw_obs is None:
            return [np.ones(s, dtype=bool) for s in self._head_sizes]
        mask_part = self._raw_obs[-self._msk_size:]
        masks: list[np.ndarray] = []
        offset = 0
        for size in self._head_sizes:
            masks.append(mask_part[offset : offset + size].astype(bool))
            offset += size
        return masks


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
    log_path = os.path.join(args.log_dir, args.run_name)
    opponent_pool = OpponentPool(max_size=args.opponent_pool_size)

    def _make_env() -> gym.Env:
        env = ContainerEnv(num_players=NUM_PLAYERS, num_colors=NUM_COLORS)
        if args.self_play:
            models = opponent_pool.sample(NUM_PLAYERS - 1, 1000.0, device=args.device)
            opponent_models = {}
            for i, m in enumerate(models):
                opponent_models[i + 1] = m
            env = SelfPlayWrapper(env, opponent_models, main_player=0)
        return ContainerMaskWrapper(env, num_players=NUM_PLAYERS, num_colors=NUM_COLORS)

    vec_env = make_vec_env(_make_env, n_envs=args.num_envs, seed=args.seed,
                           vec_env_cls=gym.vector.SyncVectorEnv)

    eval_env = make_vec_env(_make_env, n_envs=1, seed=args.seed + 1,
                            vec_env_cls=gym.vector.SyncVectorEnv)

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
