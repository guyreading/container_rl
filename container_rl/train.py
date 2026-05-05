"""PPO training script using Stable-Baselines3 with TensorBoard logging.

Usage:
    uv run python -m container_rl.train
    uv run python -m container_rl.train --num-envs 8 --total-timesteps 5000000
    uv run python -m container_rl.train --run-name my-experiment
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from container_rl import ContainerEnv

NUM_PLAYERS = 4
NUM_COLORS = 5


class SB3ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.unwrapped.observation_space.shape[0],),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent on Container environment"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="Number of steps per rollout",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs per update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="container-ppo",
        help="Run name for TensorBoard",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency (timesteps)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    args = parser.parse_args()

    log_path = os.path.join(args.log_dir, args.run_name)

    def make_env():
        env = ContainerEnv(num_players=NUM_PLAYERS, num_colors=NUM_COLORS)
        return SB3ObsWrapper(env)

    vec_env = make_vec_env(
        make_env,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=gym.vector.SyncVectorEnv,
    )

    eval_env = make_vec_env(
        make_env,
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

    model = PPO(
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
        device="cpu",
    )

    print(f"\nPPO training: {args.run_name}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Log dir: {log_path}")
    print(f"  Action space: {vec_env.action_space}")
    print(f"  Observation space shape: {vec_env.observation_space.shape}\n")

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
