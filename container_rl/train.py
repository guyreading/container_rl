"""PPO training script using Stable-Baselines3 with action masking and TensorBoard.

Usage:
    uv run python -m container_rl.train
    uv run python -m container_rl.train --num-envs 8 --total-timesteps 5000000
    uv run python -m container_rl.train --run-name my-experiment
"""

import argparse
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from container_rl import ContainerEnv
from container_rl.env.container import head_sizes, mask_size

NUM_PLAYERS = 4
NUM_COLORS = 5


class ContainerMaskWrapper(gym.ObservationWrapper):
    """Extracts per-head action masks from the observation and exposes them.

    The base observation has action masks appended as the last ``mask_size``
    elements.  This wrapper:

    * Strips the masks from the observation before passing them to the policy.
    * Exposes an ``action_masks()`` method that returns a list of boolean
      arrays (one per MultiDiscrete head) so that ``MaskablePPO`` can zero
      out invalid actions before sampling.
    """

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
        self._raw_obs: Optional[np.ndarray] = None  # stored for mask extraction

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        self._raw_obs = obs
        return obs[: -self._msk_size]

    def action_masks(self) -> list[np.ndarray]:
        """Return a list of boolean masks, one per MultiDiscrete head."""
        if self._raw_obs is None:
            return [np.ones(s, dtype=bool) for s in self._head_sizes]
        mask_part = self._raw_obs[-self._msk_size:]
        masks: list[np.ndarray] = []
        offset = 0
        for size in self._head_sizes:
            masks.append(mask_part[offset : offset + size].astype(bool))
            offset += size
        return masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on Container environment")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total timesteps to train")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="Number of steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--run-name", type=str, default="container-ppo", help="Run name for TensorBoard")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")

    args = parser.parse_args()

    log_path = os.path.join(args.log_dir, args.run_name)

    def _make_env() -> gym.Env:
        env = ContainerEnv(num_players=NUM_PLAYERS, num_colors=NUM_COLORS)
        return ContainerMaskWrapper(env, num_players=NUM_PLAYERS, num_colors=NUM_COLORS)

    vec_env = make_vec_env(
        _make_env,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=gym.vector.SyncVectorEnv,
    )

    eval_env = make_vec_env(
        _make_env,
        n_envs=1,
        seed=args.seed + 1,
        vec_env_cls=gym.vector.SyncVectorEnv,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=log_path,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=log_path,
        seed=args.seed,
        verbose=1,
        device=args.device,
    )

    print(f"\nMaskablePPO training: {args.run_name}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Log dir: {log_path}")
    print(f"  Action space: {vec_env.action_space}")
    print(f"  Observation space shape: {vec_env.observation_space.shape}")
    head_desc = " × ".join(str(s) for s in head_sizes(NUM_PLAYERS, NUM_COLORS))
    print(f"  Mask heads (MultiDiscrete): [{head_desc}]\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(os.path.join(log_path, "final_model"))

    vec_env.close()
    eval_env.close()

    print(f"\nTraining complete. Model saved to {log_path}")


if __name__ == "__main__":
    main()
