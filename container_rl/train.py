"""PPO training script for the Container RL environment with TensorBoard logging.

Usage:
    uv run python -m container_rl.train
    uv run python -m container_rl.train --num-envs 8 --total-timesteps 5000000
    uv run python -m container_rl.train --run-name my-experiment
"""

import argparse
import json
import os
import time

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tensorboard.backend.event_file.event_file_writer import EventFileWriter
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.util.tensor_util import make_tensor_proto

from container_rl import ActionEncoder, ContainerEnv

NUM_PLAYERS = 2
NUM_COLORS = 5


class FlattenObsWrapper(gym.ObservationWrapper):
    """Convert JAX observation array to numpy for gymnasium compatibility."""

    def __init__(self, env):
        super().__init__(env)
        env.reset()
        self.obs_size = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)


class FlattenActionWrapper(gym.ActionWrapper):
    """Flatten MultiDiscrete action to single discrete action index."""

    def __init__(self, env):
        super().__init__(env)
        self.encoder = ActionEncoder(
            num_players=NUM_PLAYERS, num_colors=NUM_COLORS
        )
        self.action_space = gym.spaces.Discrete(self.encoder.total_actions)

    def action(self, action):
        return self.encoder.to_multi_head(int(action))

    def step(self, action):
        return super().step(self.action(action))


def make_env(env_idx):
    def thunk():
        env = ContainerEnv(num_players=NUM_PLAYERS, num_colors=NUM_COLORS)
        env = FlattenActionWrapper(env)
        env = FlattenObsWrapper(env)
        return env

    return thunk


class ActorCritic(nn.Module):
    """Simple MLP policy network for flattened discrete actions."""

    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        action_logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x).squeeze(-1)

        return action_logits, value


class PPOTrainer:
    """PPO trainer with TensorBoard logging."""

    def __init__(
        self,
        num_envs=4,
        total_timesteps=2_000_000,
        learning_rate=2.5e-4,
        num_steps=128,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        run_name="container-ppo",
        log_dir="runs",
        seed=42,
    ):
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.run_name = run_name
        self.log_dir = log_dir
        self.seed = seed

        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_iterations = total_timesteps // self.batch_size

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(i) for i in range(num_envs)]
        )

        self.action_dim = self.envs.single_action_space.n
        self.obs_shape = self.envs.single_observation_space.shape

        self.key = jax.random.PRNGKey(seed)

        self.network = ActorCritic(action_dim=self.action_dim)

        dummy_obs = self.envs.observation_space.sample()
        self.params = self.network.init(self.key, dummy_obs)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )
        self.opt_state = self.optimizer.init(self.params)

        self.log_path = os.path.join(log_dir, run_name)
        os.makedirs(self.log_path, exist_ok=True)

        self.tb_writer = EventFileWriter(self.log_path)

        self._log_hparams()

    def _log_hparams(self):
        hparams = {
            "num_envs": self.num_envs,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            "clip_coef": self.clip_coef,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "action_dim": self.action_dim,
            "obs_shape": str(self.obs_shape),
            "batch_size": self.batch_size,
            "num_iterations": self.num_iterations,
        }
        with open(os.path.join(self.log_path, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)

    def log_scalar(self, tag, value, step):
        summary = Summary(value=[
            Summary.Value(tag=tag, tensor=make_tensor_proto(float(value)))
        ])
        event = Event(summary=summary, step=step, wall_time=time.time())
        self.tb_writer.add_event(event)

    @jax.jit
    def get_actions_and_values(self, params, obs):
        action_logits, value = self.network.apply(params, obs)
        return action_logits, value

    @jax.jit
    def compute_loss(
        self,
        params,
        obs,
        actions,
        logprobs,
        advantages,
        returns,
        values,
    ):
        action_logits, new_values = self.network.apply(params, obs)

        logprobs_new = jax.nn.log_softmax(action_logits)
        logprob = jnp.take_along_axis(
            logprobs_new, jnp.expand_dims(actions, -1), axis=-1
        ).squeeze(-1)

        logratio = logprob - logprobs
        ratio = jnp.exp(logratio)

        approx_kl = ((ratio - 1) - logratio).mean()

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        new_values = new_values.reshape(-1)
        v_loss_unclipped = (new_values - returns) ** 2
        clipped_values = values + jnp.clip(
            new_values - values, -self.clip_coef, self.clip_coef
        )
        v_loss_clipped = (clipped_values - returns) ** 2
        v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

        entropy = -jnp.sum(
            jax.nn.softmax(action_logits) * logprobs_new, axis=-1
        ).mean()

        loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy
        return loss, (pg_loss, v_loss, entropy, approx_kl, new_values.mean())

    @jax.jit
    def update_step(
        self,
        params,
        opt_state,
        obs,
        actions,
        logprobs,
        advantages,
        returns,
        values,
    ):
        grad_fn = jax.grad(self.compute_loss, has_aux=True)
        grads, aux = grad_fn(
            params, obs, actions, logprobs, advantages, returns, values
        )
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, aux

    def train(self):
        print(f"Starting PPO training: {self.run_name}")
        print(f"  Environments: {self.num_envs}")
        print(f"  Total timesteps: {self.total_timesteps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Minibatch size: {self.minibatch_size}")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Log dir: {self.log_path}")

        obs = np.array(self.envs.reset()[0])
        num_actions = self.action_dim

        rollout_obs = np.zeros(
            (self.num_steps, self.num_envs) + self.obs_shape, dtype=np.float32
        )
        rollout_actions = np.zeros(
            (self.num_steps, self.num_envs), dtype=np.int32
        )
        rollout_logprobs = np.zeros(
            (self.num_steps, self.num_envs), dtype=np.float32
        )
        rollout_rewards = np.zeros(
            (self.num_steps, self.num_envs), dtype=np.float32
        )
        rollout_dones = np.zeros(
            (self.num_steps, self.num_envs), dtype=np.float32
        )
        rollout_values = np.zeros(
            (self.num_steps, self.num_envs), dtype=np.float32
        )

        global_step = 0
        start_time = time.time()
        current_lr = self.learning_rate

        for iteration in range(1, self.num_iterations + 1):
            iter_start = time.time()

            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                current_lr = self.learning_rate * frac
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(learning_rate=current_lr, eps=1e-5),
                )

            for step in range(self.num_steps):
                global_step += 1

                rollout_obs[step] = obs
                obs_jax = jnp.array(obs)

                action_logits, values = self.get_actions_and_values(
                    self.params, obs_jax
                )
                values = np.array(values)
                rollout_values[step] = values

                probs = jax.nn.softmax(np.array(action_logits))
                log_probs = jax.nn.log_softmax(np.array(action_logits))

                actions = np.zeros(self.num_envs, dtype=np.int32)
                logprob_vals = np.zeros(self.num_envs, dtype=np.float32)

                for e in range(self.num_envs):
                    action = np.random.choice(num_actions, p=probs[e])
                    actions[e] = action
                    logprob_vals[e] = log_probs[e, action]

                rollout_actions[step] = actions
                rollout_logprobs[step] = logprob_vals

                obs, rewards, terminated, truncated, infos = self.envs.step(
                    actions
                )
                obs = np.array(obs)
                done = np.logical_or(terminated, truncated).astype(np.float32)
                rollout_rewards[step] = rewards
                rollout_dones[step] = done

                for e in range(self.num_envs):
                    if terminated[e]:
                        if "episode" in infos and "r" in infos["episode"]:
                            ep_reward = float(infos["episode"]["r"][e])
                            ep_len = int(infos["episode"]["l"][e])
                            print(
                                f"  Episode reward: {ep_reward:.2f}, "
                                f"length: {ep_len}, "
                                f"step: {global_step}"
                            )

            with jax.default_device(jax.devices("cpu")[0]):
                next_obs_jax = jnp.array(obs)
                _, last_values = self.get_actions_and_values(
                    self.params, next_obs_jax
                )
                last_values = np.array(last_values).flatten()

            advantages = np.zeros_like(rollout_rewards)
            last_gae_lam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - rollout_dones[t]
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - rollout_dones[t + 1]
                    next_values = rollout_values[t + 1]

                delta = (
                    rollout_rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - rollout_values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )

            returns = advantages + rollout_values

            b_obs = rollout_obs.reshape((-1,) + self.obs_shape)
            b_actions = rollout_actions.reshape(-1)
            b_logprobs = rollout_logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = rollout_values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clip_fracs = []

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    (
                        self.params,
                        self.opt_state,
                        (pg_loss, v_loss, entropy, approx_kl, v_pred),
                    ) = self.update_step(
                        self.params,
                        self.opt_state,
                        jnp.array(b_obs[mb_inds]),
                        jnp.array(b_actions[mb_inds]),
                        jnp.array(b_logprobs[mb_inds]),
                        jnp.array(b_advantages[mb_inds]),
                        jnp.array(b_returns[mb_inds]),
                        jnp.array(b_values[mb_inds]),
                    )

                    clip_fracs.append(
                        float(
                            jnp.mean(
                                jnp.abs(
                                    (
                                        jnp.exp(
                                            jnp.take_along_axis(
                                                jax.nn.log_softmax(
                                                    self.network.apply(
                                                        self.params,
                                                        jnp.array(b_obs[mb_inds]),
                                                    )[0],
                                                ),
                                                jnp.expand_dims(
                                                    jnp.array(b_actions[mb_inds]), -1
                                                ),
                                                axis=-1,
                                            ).squeeze(-1)
                                        )
                                        / jnp.array(b_logprobs[mb_inds])
                                        - 1
                                    )
                                )
                                > self.clip_coef
                            )
                        )
                    )

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

            y_pred = b_values.flatten()
            y_true = b_returns.flatten()
            var_y = np.var(y_true)
            explained_var = (
                float(1 - np.var(y_true - y_pred) / var_y) if var_y > 0 else 0
            )

            iter_time = time.time() - iter_start
            sps = int(global_step / (time.time() - start_time))

            print(
                f"  Iteration {iteration}/{self.num_iterations}, "
                f"SPS: {sps}, "
                f"Loss: pg={float(pg_loss):.3f} vf={float(v_loss):.3f} "
                f"ent={float(entropy):.3f}, "
                f"approx_kl={float(approx_kl):.4f}, "
                f"explained_var={explained_var:.3f}"
            )

            self.log_scalar("charts/SPS", sps, global_step)
            self.log_scalar("losses/policy_loss", float(pg_loss), global_step)
            self.log_scalar("losses/value_loss", float(v_loss), global_step)
            self.log_scalar("losses/entropy", float(entropy), global_step)
            self.log_scalar("losses/approx_kl", float(approx_kl), global_step)
            self.log_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            self.log_scalar(
                "losses/clip_fraction",
                float(np.mean(clip_fracs)) if clip_fracs else 0,
                global_step,
            )
            self.log_scalar(
                "charts/learning_rate",
                float(current_lr),
                global_step,
            )

        self.tb_writer.flush()
        self.tb_writer.close()

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Final SPS: {int(global_step / total_time)}")
        print(f"Logs saved to: {self.log_path}")

        self.envs.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Container environment")
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
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
        "--num-steps",
        type=int,
        default=128,
        help="Number of steps per rollout",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
        help="Number of minibatches per update",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="Number of epochs to update on each batch",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="PPO clip coefficient",
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
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="container-ppo",
        help="Name for this run (used in TensorBoard log dir)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--no-anneal-lr",
        action="store_true",
        help="Disable learning rate annealing",
    )

    args = parser.parse_args()

    trainer = PPOTrainer(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        anneal_lr=not args.no_anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        run_name=args.run_name,
        log_dir=args.log_dir,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()
