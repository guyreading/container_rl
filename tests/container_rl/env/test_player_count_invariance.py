"""Observation and action shapes must not depend on the player count.

A policy network has a fixed input and output width, so if a 3-player game
produced a shorter observation or a narrower opponent head than a 5-player
game, a model trained at one count could not even be loaded at another.
The env therefore always lays out ``MAX_PLAYERS`` seats:

* **Observation** — one identically-shaped block per seat (seat 0 is the
  acting player, the rest follow clockwise), then shared game state, then the
  action masks.  Seats nobody occupies are filled with ``NULL_OBS`` and the
  game-state block carries a ``seat_present`` flag per seat.
* **Action** — the opponent head always has a slot for each of the four other
  seats.  Slots for absent seats are masked in every mode, so they can never be
  chosen.

These tests pin both halves, and the alignment between them: opponent index
*j* and observation seat *j* must always name the same player, or the policy
would learn to read one player's stock and buy from another.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_disable_jit", True)

from container_rl.env.container import (
    ACTION_BUY_FROM_FACTORY_STORE,
    HEAD_ACTION_TYPE,
    HEAD_OPPONENT,
    LOCATION_HARBOUR_OFFSET,
    MAX_PLAYERS,
    NULL_OBS,
    SECRET_CARD_VALUES,
    ContainerFunctional,
    game_obs_size,
    head_sizes,
    mask_size,
    seat_obs_size,
)

NC = 5
PLAYER_COUNTS = (2, 3, 4, 5)
MASK_HEAD_ORDER = ("action_type", "opponent", "color", "price_slot", "purchase")
SEAT = seat_obs_size(NC)

# Offsets of the scalar features inside one seat block (see ``observation``).
CASH, LOANS, WAREHOUSES, SHIP_LOCATION = 0, 1, 2, 3


def _env(num_players):
    return ContainerFunctional(num_players=num_players, num_colors=NC)


def _initial(env, **overrides):
    return env.initial(jax.random.PRNGKey(0), env.params)._replace(**overrides)


def _obs(env, state):
    return np.asarray(env.observation(state, jax.random.PRNGKey(0), env.params))


def _seats(obs):
    return obs[: MAX_PLAYERS * SEAT].reshape(MAX_PLAYERS, SEAT)


def _game(obs):
    start = MAX_PLAYERS * SEAT
    return obs[start : start + game_obs_size(NC)]


def _distinct_cash(num_players):
    """Cash of 11, 12, … so every seat block is identifiable by its cash."""
    return jnp.arange(11, 11 + num_players, dtype=jnp.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Shapes
# ══════════════════════════════════════════════════════════════════════════════


def test_spaces_identical_across_player_counts():
    """Every player count must declare the same observation and action spaces.

    **Why**: this is the whole point — one checkpoint has to load and act at
    any table size.  A count-dependent width fails at ``model.predict``.
    """
    envs = [_env(n) for n in PLAYER_COUNTS]
    obs_shapes = {e.observation_space.shape for e in envs}
    action_nvecs = {tuple(int(x) for x in e.action_space.nvec) for e in envs}
    assert len(obs_shapes) == 1, f"observation shapes differ: {obs_shapes}"
    assert len(action_nvecs) == 1, f"action heads differ: {action_nvecs}"


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_observation_fills_declared_space(num_players):
    """The assembled observation must be exactly the declared width.

    **Why**: the masks are sliced off the tail, so any length drift shifts
    them onto the wrong heads (see ``test_mask_packing.py``).
    """
    env = _env(num_players)
    assert _obs(env, _initial(env)).shape == env.observation_space.shape


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_opponent_head_has_a_slot_for_every_other_seat(num_players):
    """The opponent head is no-op plus four seats, whatever the count."""
    assert head_sizes(num_players, NC)[HEAD_OPPONENT] == MAX_PLAYERS


@pytest.mark.parametrize("num_players", [0, 1, MAX_PLAYERS + 1])
def test_unsupported_player_counts_are_rejected(num_players):
    """Counts the layout cannot represent must fail at construction.

    **Why**: a sixth player has no seat block and no opponent slot.  Without
    the check the env would build and then fail deep inside an observation,
    or worse, silently drop a player from what the policy sees.
    """
    with pytest.raises(ValueError, match="num_players"):
        _env(num_players)


# ══════════════════════════════════════════════════════════════════════════════
# Null seats
# ══════════════════════════════════════════════════════════════════════════════


def test_null_is_distinct_from_real_readings():
    """``NULL_OBS`` must not be 0 or a secret-card value.

    **Why**: 0 is a genuine reading for almost every per-seat feature (no
    loans, no stock, ship at open sea), and -1 is the 5/10 secret card.  A
    null that looks like either tells the policy an absent seat is a real,
    empty-handed player.
    """
    assert NULL_OBS != 0
    assert NULL_OBS not in SECRET_CARD_VALUES


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_absent_seats_are_null_and_present_seats_are_not(num_players):
    """Every element of an unoccupied seat is ``NULL_OBS``; occupied seats are real."""
    env = _env(num_players)
    seats = _seats(_obs(env, _initial(env)))
    for k in range(num_players):
        assert not np.all(seats[k] == NULL_OBS), f"occupied seat {k} reads as null"
        assert seats[k][CASH] > 0, f"occupied seat {k} lost its cash"
    for k in range(num_players, MAX_PLAYERS):
        assert np.all(seats[k] == NULL_OBS), f"absent seat {k} is not fully null"


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_seat_present_flags(num_players):
    """The game-state block opens with one presence flag per seat.

    **Why**: no sentinel is collision-proof on its own — cash can go negative
    after a rejected auction — so the flags are the unambiguous signal.
    """
    env = _env(num_players)
    flags = _game(_obs(env, _initial(env)))[:MAX_PLAYERS]
    expected = [1.0] * num_players + [0.0] * (MAX_PLAYERS - num_players)
    np.testing.assert_array_equal(flags, expected)


# ══════════════════════════════════════════════════════════════════════════════
# Seat order and its alignment with the opponent head
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_seat_zero_is_acting_player_then_clockwise(num_players):
    """Seat *k* holds player ``(current_player + k) % num_players``.

    **Why**: an ego-centric layout lets one set of weights play from any seat.
    The acting player is deliberately not player 0 here so the rotation is
    actually exercised.
    """
    env = _env(num_players)
    cp = num_players - 1
    cash = _distinct_cash(num_players)
    seats = _seats(_obs(env, _initial(env, cash=cash,
                                      current_player=jnp.array(cp, dtype=jnp.int32))))
    for k in range(num_players):
        assert seats[k][CASH] == int(cash[(cp + k) % num_players]), f"seat {k} is the wrong player"


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_opponent_index_and_observation_seat_name_the_same_player(num_players):
    """Choosing opponent *j* must target the player shown in seat *j*.

    **Why**: the policy decides whom to buy from by reading the seat blocks.
    If the head counted players in a different order from the observation, it
    would inspect one player's store and buy from another's.  This drives the
    real handler rather than ``_get_target_player`` so the clamp inside it is
    covered too.
    """
    env = _env(num_players)
    cp = 1
    state = _initial(env, cash=_distinct_cash(num_players),
                     current_player=jnp.array(cp, dtype=jnp.int32))
    seats = _seats(_obs(env, state))
    for j in range(1, num_players):
        action = (jnp.zeros(5, dtype=jnp.int32)
                  .at[HEAD_ACTION_TYPE].set(ACTION_BUY_FROM_FACTORY_STORE + 1)
                  .at[HEAD_OPPONENT].set(j))
        after = env._action_buy_from_factory_store(state, action, env.params)
        assert int(after.shopping_active) == 1, f"opponent {j} was not a valid target"
        target = int(after.shopping_target)
        assert seats[j][CASH] == int(state.cash[target]), (
            f"opponent head {j} bought from player {target}, but seat {j} shows someone else"
        )


def test_harbour_locations_are_seat_relative():
    """A ship in a harbour must name that harbour's owner by seat, not absolute index.

    **Why**: every other player reference in the observation is rotated.  An
    absolute harbour index would mean "seat 2's harbour" to the policy while
    actually pointing at whoever sits at absolute index 2.
    """
    env = _env(5)
    cp = 3
    state = _initial(
        env,
        current_player=jnp.array(cp, dtype=jnp.int32),
        # player 3 (seat 0) docked at player 1's harbour; player 4 (seat 1) at player 0's
        ship_location=jnp.array([0, 0, 0,
                                 LOCATION_HARBOUR_OFFSET + 1,
                                 LOCATION_HARBOUR_OFFSET + 0], dtype=jnp.int32),
    )
    seats = _seats(_obs(env, state))
    assert seats[0][SHIP_LOCATION] == LOCATION_HARBOUR_OFFSET + (1 - cp) % 5  # seat 3
    assert seats[1][SHIP_LOCATION] == LOCATION_HARBOUR_OFFSET + (0 - cp) % 5  # seat 2
    assert seats[2][SHIP_LOCATION] == 0, "open sea must stay open sea"


# ══════════════════════════════════════════════════════════════════════════════
# Absent opponents are masked in every mode
# ══════════════════════════════════════════════════════════════════════════════


def _parallel(env):
    return _initial(env)


def _shopping(env):
    return _initial(env,
                    shopping_active=jnp.array(1, dtype=jnp.int32),
                    shopping_action_type=jnp.array(ACTION_BUY_FROM_FACTORY_STORE, dtype=jnp.int32),
                    shopping_target=jnp.array(1, dtype=jnp.int32))


def _produce(env):
    state = _initial(env, produce_active=jnp.array(1, dtype=jnp.int32))
    return state._replace(produce_pending=state.produce_pending.at[0].set(1))


def _auction(env):
    state = _initial(env,
                     auction_active=jnp.array(1, dtype=jnp.int32),
                     auction_seller=jnp.array(0, dtype=jnp.int32))
    return state._replace(auction_cargo=state.auction_cargo.at[0].set(1))


MODES = [("parallel", _parallel), ("shopping", _shopping),
         ("produce", _produce), ("auction", _auction)]


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
@pytest.mark.parametrize("mode,builder", MODES)
def test_absent_opponent_slots_are_masked(num_players, mode, builder):
    """No mode may ever offer an opponent slot past the last player.

    **Why**: in a 3-player game slots 3 and 4 of the opponent head point at
    nobody.  The handlers clamp an out-of-range index onto a real player, so
    if the mask let one through the policy would silently target the wrong
    seat instead of being refused.  Auction mode is included because it
    repurposes the head as an absolute bidder index.
    """
    env = _env(num_players)
    opp = np.asarray(env._action_masks(builder(env), env.params)["opponent"])
    assert opp.shape == (MAX_PLAYERS,)
    assert not opp[num_players:].any(), (
        f"{mode}: absent seats offered on the opponent head: {opp.tolist()}"
    )


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_present_opponents_stay_selectable(num_players):
    """Masking absent seats must not mask real ones.

    **Why**: the opposite failure — every player starts with a $2 container
    in their factory store, so each real opponent is a legal target and all of
    them must be offered, including the one in the last occupied seat.
    """
    env = _env(num_players)
    opp = np.asarray(env._action_masks(_parallel(env), env.params)["opponent"])
    assert opp[1:num_players].all(), f"a present opponent is masked: {opp.tolist()}"


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_auction_bidder_slots_cover_exactly_the_players(num_players):
    """During an auction the head names bidders by absolute index 0..np-1."""
    env = _env(num_players)
    opp = np.asarray(env._action_masks(_auction(env), env.params)["opponent"])
    expected = [1] * num_players + [0] * (MAX_PLAYERS - num_players)
    np.testing.assert_array_equal(opp, expected)


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_packed_masks_match_computed_masks(num_players):
    """The mask tail of the observation must equal ``_action_masks`` at every count."""
    env = _env(num_players)
    state = _initial(env)
    masks = env._action_masks(state, env.params)
    expected = np.concatenate([np.asarray(masks[k], dtype=np.float32) for k in MASK_HEAD_ORDER])
    np.testing.assert_array_equal(_obs(env, state)[-mask_size(num_players, NC):], expected)
