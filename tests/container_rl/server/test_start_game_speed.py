"""Starting a game must not pay for work nobody reads.

``maybe_start_game`` used to call ``env.reset()``, which builds the policy
observation on top of the state -- and the bulk of that is ``_action_masks``,
the most expensive thing the env does.  The server throws the observation
away: it serialises the state and computes masks later, per request, for
whoever is actually to move.  ``reset_state_only`` skips it.

The thing worth pinning is not the speed but the equivalence: the state the
shortcut leaves behind, and the rng it leaves the env on, have to match what
``reset()`` would have produced, or games would diverge from their seed.
"""

from __future__ import annotations

import jax.tree_util as jtu
import numpy as np
import pytest

from container_rl.env.container import ContainerJaxEnv
from container_rl.server.database import Database
from container_rl.server.game_manager import GameManager


def leaves(state):
    return [np.asarray(x) for x in jtu.tree_leaves(state)]


def same(a, b):
    la, lb = leaves(a), leaves(b)
    return len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb))


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_state_only_reset_matches_full_reset(num_players, seed):
    full = ContainerJaxEnv(num_players=num_players, num_colors=5)
    full.reset(seed=seed)
    quick = ContainerJaxEnv(num_players=num_players, num_colors=5)
    returned = quick.reset_state_only(seed=seed)

    assert same(full.state, quick.state)
    assert same(full.state, returned)
    # The rng has to advance identically too, or the first step diverges.
    assert np.array_equal(np.asarray(full.rng), np.asarray(quick.rng))


def test_started_game_state_matches_a_full_reset(tmp_path):
    """The state the manager saves is the one the seed says it should be."""
    db = Database(str(tmp_path / "games.db"))
    manager = GameManager(db, lambda *a, **k: None)
    res = manager.create_game_trusted("alice", 4, 5, seed=99, ai_count=3)
    assert manager.maybe_start_game(res["game_id"])

    expected = ContainerJaxEnv(num_players=4, num_colors=5)
    expected.reset(seed=99)
    assert same(manager.get_state(res["game_id"]), expected.state)


def test_warm_up_leaves_no_games_behind(tmp_path):
    """Warm-up builds throwaway envs; it must not register any of them."""
    db = Database(str(tmp_path / "games.db"))
    manager = GameManager(db, lambda *a, **k: None)
    manager.warm_up()
    assert manager._envs == {}
    assert db.list_joinable_games() == []
