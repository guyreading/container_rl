"""Tests for how ``train.py`` sets up the games the agent learns from.

Two properties matter here:

* **Player-count rotation.**  Observation and action shapes are the same at
  every table size, but a JAX env cannot be resized in place, so
  ``PlayerCountCycler`` keeps one env per count and moves to the next count on
  every episode reset.  If the rotation stalled, the policy would silently train
  at one table size only.
* **Every seat is the agent.**  Outside ``--self-play`` nothing substitutes
  scripted or frozen opponents: each step's observation is centred on the seat
  whose decision it is, and its reward is that seat's net-worth change.  If the
  reward were pinned to one seat, the policy would learn to play the other
  seats in that seat's favour.
"""

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from container_rl.env.container import (
    ACTION_PASS,
    HEAD_ACTION_TYPE,
    HEAD_OPPONENT,
    MAX_PLAYERS,
    head_sizes,
)
from container_rl.train import (
    DEFAULT_PLAYER_COUNTS,
    PlayerCountCycler,
    make_cycling_env,
    make_env,
)

OPP = slice(sum(head_sizes(MAX_PLAYERS, 5)[:HEAD_OPPONENT]),
            sum(head_sizes(MAX_PLAYERS, 5)[:HEAD_OPPONENT + 1]))


def test_default_rotation_is_three_to_five_players():
    assert tuple(DEFAULT_PLAYER_COUNTS) == (3, 4, 5)


# ══════════════════════════════════════════════════════════════════════════════
# Rotation, with lightweight stand-in envs
# ══════════════════════════════════════════════════════════════════════════════


class _OneStepGame(gym.Env):
    """Ends every episode after one step, so auto-reset can be driven quickly."""

    def __init__(self, num_players, obs_width=4):
        self.num_players = num_players
        self.observation_space = gym.spaces.Box(-1, 1, (obs_width,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.seeds = []

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        return np.full(4, self.num_players / 10, np.float32), {}

    def step(self, action):
        return np.zeros(4, np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.arange(self.num_players) >= 0


def test_resets_walk_the_counts_in_order_and_wrap():
    cycler = PlayerCountCycler([3, 4, 5], _OneStepGame)
    seen = [cycler.reset()[1]["num_players"] for _ in range(7)]
    assert seen == [3, 4, 5, 3, 4, 5, 3]


def test_start_offsets_the_first_count():
    """Envs are staggered so a vector of them spans table sizes at once."""
    firsts = [PlayerCountCycler([3, 4, 5], _OneStepGame, start=i).reset()[1]["num_players"]
              for i in range(4)]
    assert firsts == [3, 4, 5, 3]


def test_step_reports_and_masks_follow_the_active_count():
    cycler = PlayerCountCycler([3, 5], _OneStepGame)
    for expected in (3, 5, 3):
        cycler.reset()
        assert cycler.num_players == expected
        assert len(cycler.action_masks()) == expected, "masks came from a different count's env"
        assert cycler.step(0)[4]["num_players"] == expected


def test_vec_env_auto_reset_advances_the_count():
    """Finishing a game inside a VecEnv must move that env to the next count.

    **Why**: sb3 never calls ``reset`` directly mid-training — it auto-resets on
    ``done``.  The rotation only happens if that path reaches the cycler, and
    masks must still be reachable through ``Monitor`` via ``env_method``.
    """
    vec = DummyVecEnv([lambda: _monitored_cycler([3, 4, 5])])
    vec.reset()
    counts = []
    for _ in range(5):
        counts.append(vec.env_method("action_masks")[0].size)
        vec.step(np.array([0]))
    assert counts == [3, 4, 5, 3, 4]


def _monitored_cycler(counts):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(PlayerCountCycler(counts, _OneStepGame))


def test_seed_is_spread_to_every_count():
    """Each count gets its own seed the first time it plays, not just the first."""
    cycler = PlayerCountCycler([3, 4, 5], _OneStepGame)
    cycler.reset(seed=100)
    cycler.reset()
    cycler.reset()
    cycler.reset()
    assert [e.seeds[0] for e in cycler._envs] == [100, 101, 102]
    assert cycler._envs[0].seeds[1] is None, "a count must not be reseeded on its second game"


def test_mismatched_spaces_are_rejected():
    """A count whose spaces differ cannot share a policy — fail at construction."""
    with pytest.raises(ValueError, match="spaces differ"):
        PlayerCountCycler([3, 4], lambda n: _OneStepGame(n, obs_width=4 if n == 3 else 5))


def test_empty_rotation_is_rejected():
    with pytest.raises(ValueError):
        PlayerCountCycler([], _OneStepGame)


# ══════════════════════════════════════════════════════════════════════════════
# The real training envs
# ══════════════════════════════════════════════════════════════════════════════


def test_real_envs_rotate_with_matching_opponent_masks():
    """The production factory must rotate real games, masks sized to each table.

    Every player starts with a $2 container for sale, so each real opponent is
    a legal target: a 3-player game offers opponent slots 1-2, a 5-player game
    slots 1-4.
    """
    env = make_cycling_env([3, 4, 5])
    for n in (3, 4, 5, 3):
        _, info = env.reset()
        assert info["num_players"] == n
        assert env.unwrapped.active_env.unwrapped.func_env.params.num_players == n
        opp = env.unwrapped.action_masks()[OPP]
        np.testing.assert_array_equal(opp[1:], [1] * (n - 1) + [0] * (MAX_PLAYERS - n))
    env.close()


def _pass():
    a = np.zeros(5, dtype=np.int64)
    a[HEAD_ACTION_TYPE] = ACTION_PASS + 1
    return a


@pytest.mark.parametrize("num_players", DEFAULT_PLAYER_COUNTS)
def test_policy_is_asked_to_act_for_every_seat(num_players):
    """Play passes round the table: each seat's decision comes back to the policy.

    **Why**: this is what "every player is the agent" means mechanically.  With
    no opponent wrapper, every ``step`` returns control to the learner, and the
    observation it gets back is centred on the new acting seat (seat 0 shows that
    player's cash).
    """
    env = make_env(num_players)
    env.reset(seed=0)
    base = env.unwrapped
    base.state = base.state._replace(cash=jnp.arange(11, 11 + num_players, dtype=jnp.int32))
    actors = []
    for _ in range(2 * num_players):  # two actions per turn
        obs, *_ = env.step(_pass())
        cp = int(base.state.current_player)
        actors.append(cp)
        assert obs[0] == float(base.state.cash[cp]), "observation is not centred on the acting seat"
    assert set(actors) == set(range(num_players)), f"some seat never reached the policy: {actors}"


def test_reward_is_paid_to_the_seat_that_acted():
    """Reward is the acting seat's net-worth change, not a fixed seat's.

    **Why**: one policy plays all seats, so a reward pinned to seat 0 would
    teach it to throw seats 1-4's games to seat 0.
    """
    env = make_env(3)
    env.reset(seed=0)
    func = env.unwrapped.func_env
    state = env.unwrapped.state._replace(current_player=jnp.array(2, dtype=jnp.int32))
    next_state = state._replace(cash=state.cash.at[2].add(5).at[0].add(-3))
    assert float(func.reward(state, None, next_state, None)) == 5.0
