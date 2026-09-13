"""Tests that the action masks survive the trip into the observation.

``_action_masks`` is well covered by ``test_action_space.py``, but nothing
checked the *packing*: the masks are appended to the tail of the
observation vector and the training wrapper recovers them by slicing
``obs[-mask_size:]``.  That slice is only correct if the assembled
observation is exactly as long as ``observation_space`` declares.

It was not.  The ``obs_size`` arithmetic in ``ContainerFunctional.__init__``
budgeted 5 floats for the auction block while ``observation`` appends 3, so
``observation`` zero-padded the tail by 2 and every head's mask was read
two positions to the left of where it was written.  ``MOVE_AUCTION`` ended
up gated by ``TAKE_LOAN``'s flag, ``PURCHASE_STOP`` became permanently
unselectable, and the per-head no-op was never masked in parallel mode.

These tests pin the packing itself, so a future change to the observation
layout that forgets to update ``obs_size`` fails here rather than quietly
mistraining the policy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_disable_jit", True)

from container_rl.env.container import (
    ACTION_BUY_FACTORY,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_DOMESTIC_SALE,
    ACTION_MOVE_AUCTION,
    ACTION_MOVE_LOAD,
    ACTION_PASS,
    ACTION_TAKE_LOAN,
    LOCATION_OPEN_SEA,
    PURCHASE_STOP,
    ContainerFunctional,
    head_sizes,
    mask_size,
)

from tests.container_rl.env.test_action_space import _make_params, _make_state

MASK_HEAD_ORDER = ("action_type", "opponent", "color", "price_slot", "purchase")


def _func_env(num_players=2, num_colors=5):
    return ContainerFunctional(num_players=num_players, num_colors=num_colors)


def _obs(func_env, state, params):
    return np.asarray(
        func_env.observation(state, jax.random.PRNGKey(0), params), dtype=np.float32
    )


def _true_mask(func_env, state, params):
    """The masks as ``_action_masks`` computes them, concatenated in head order."""
    masks = func_env._action_masks(state, params)
    return np.concatenate(
        [np.asarray(masks[k], dtype=np.float32) for k in MASK_HEAD_ORDER]
    )


def _packed_mask(func_env, state, params):
    """The masks as a consumer recovers them from the observation tail."""
    return _obs(func_env, state, params)[-mask_size(params.num_players, params.num_colors):]


def _head_slice(name, num_players=2, num_colors=5):
    sizes = head_sizes(num_players, num_colors)
    start = int(np.cumsum([0] + sizes)[MASK_HEAD_ORDER.index(name)])
    return slice(start, start + sizes[MASK_HEAD_ORDER.index(name)])


# ══════════════════════════════════════════════════════════════════════════════
# Layout invariants
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_players,num_colors", [(2, 5), (3, 5), (4, 5), (4, 4)])
def test_observation_length_matches_declared_space(num_players, num_colors):
    """``observation`` must fill ``observation_space`` exactly.

    **Why**: the masks sit at the tail, so a short observation gets padded
    behind them and a long one has them truncated.  Either way the
    consumer's ``obs[-mask_size:]`` slice stops lining up with the heads.
    Equality here is what makes the tail slice meaningful.
    """
    func_env = _func_env(num_players, num_colors)
    params = _make_params(num_players, num_colors)
    state = _make_state() if (num_players, num_colors) == (2, 5) else func_env.initial(
        jax.random.PRNGKey(0), params
    )
    obs = _obs(func_env, state, params)
    assert obs.shape[0] == func_env.observation_space.shape[0]


def test_observation_is_not_silently_padded():
    """A wrong ``obs_size`` must raise, not pad.

    **Why**: silent padding is how the original misalignment hid — the
    shapes agreed, the tests passed, and only the policy saw the damage.
    """
    func_env = _func_env()
    params = _make_params()
    state = _make_state()
    from gymnasium import spaces

    func_env.observation_space = spaces.Box(
        low=0, high=100,
        shape=(func_env.observation_space.shape[0] + 2,), dtype=np.float32,
    )
    with pytest.raises(AssertionError, match="misaligned"):
        func_env.observation(state, jax.random.PRNGKey(0), params)


# ══════════════════════════════════════════════════════════════════════════════
# Packed mask == computed mask, across every mode
# ══════════════════════════════════════════════════════════════════════════════


def _state_parallel():
    return _make_state()


def _state_auction_legal():
    """P0 at open sea with cargo — ``MOVE_AUCTION`` is genuinely legal."""
    state = _make_state()
    return state._replace(
        ship_location=state.ship_location.at[0].set(LOCATION_OPEN_SEA),
        ship_contents=state.ship_contents.at[0, 0].set(1),
    )


def _state_auction_active():
    state = _state_auction_legal()
    return state._replace(
        auction_active=jnp.array(1, dtype=jnp.int32),
        auction_seller=jnp.array(0, dtype=jnp.int32),
        auction_cargo=state.auction_cargo.at[0].set(1),
        current_player=jnp.array(1, dtype=jnp.int32),
    )


def _state_shopping():
    state = _make_state()
    return state._replace(
        shopping_active=jnp.array(1, dtype=jnp.int32),
        shopping_action_type=jnp.array(ACTION_BUY_FROM_FACTORY_STORE, dtype=jnp.int32),
        shopping_target=jnp.array(1, dtype=jnp.int32),
    )


def _state_produce():
    state = _make_state()
    return state._replace(
        produce_active=jnp.array(1, dtype=jnp.int32),
        produce_pending=state.produce_pending.at[0].set(1),
    )


@pytest.mark.parametrize(
    "name,builder",
    [
        ("parallel", _state_parallel),
        ("auction_legal", _state_auction_legal),
        ("auction_active", _state_auction_active),
        ("shopping", _state_shopping),
        ("produce", _state_produce),
    ],
)
def test_packed_mask_matches_computed_mask(name, builder):
    """Every head must arrive intact, in every game mode.

    **Why**: the misalignment was a constant two-position shift, so it
    corrupted all five heads in all modes at once.  Comparing the whole
    concatenated block catches a shift of any size, and the modes are
    enumerated because each one lights up a different set of heads.
    """
    func_env = _func_env()
    params = _make_params()
    state = builder()
    np.testing.assert_array_equal(
        _packed_mask(func_env, state, params),
        _true_mask(func_env, state, params),
        err_msg=f"packed mask differs from _action_masks in {name} mode",
    )


# ══════════════════════════════════════════════════════════════════════════════
# The specific symptoms the shift produced
# ══════════════════════════════════════════════════════════════════════════════


def test_auction_slot_tracks_auction_legality():
    """The policy's ``MOVE_AUCTION`` slot must follow auction legality.

    **Why**: this is the symptom that surfaced the bug — agents never
    chose auction.  Under the shift the slot was gated by ``TAKE_LOAN``
    (legal almost always), so the policy was offered auction in states
    where it did nothing and learned to avoid the action outright.
    """
    func_env = _func_env()
    params = _make_params()
    at = _head_slice("action_type")

    illegal = _packed_mask(func_env, _make_state(), params)[at]
    assert int(illegal[ACTION_MOVE_AUCTION + 1]) == 0, (
        "auction offered with the ship in harbour and no cargo"
    )

    legal = _packed_mask(func_env, _state_auction_legal(), params)[at]
    assert int(legal[ACTION_MOVE_AUCTION + 1]) == 1, (
        "auction withheld at open sea with cargo"
    )


def test_auction_slot_is_not_gated_by_take_loan():
    """Auction availability must not move with loan availability.

    **Why**: pins the exact off-by-two.  With two loans already taken
    ``TAKE_LOAN`` is masked; if auction is still offered at sea then the
    two flags are independent, as they should be.
    """
    func_env = _func_env()
    params = _make_params()
    at = _head_slice("action_type")

    state = _state_auction_legal()
    maxed = state._replace(loans=state.loans.at[0].set(2))
    packed = _packed_mask(func_env, maxed, params)[at]

    assert int(packed[ACTION_TAKE_LOAN + 1]) == 0, "take-loan should be masked at 2 loans"
    assert int(packed[ACTION_MOVE_AUCTION + 1]) == 1, (
        "auction availability is still tied to take-loan availability"
    )


def test_purchase_stop_is_selectable_while_shopping():
    """``PURCHASE_STOP`` is the last slot of the last head — the shift ate it.

    **Why**: STOP ends a shopping continuation.  Padding made the final
    two purchase slots read as permanent zeros, so the agent could never
    voluntarily stop buying.
    """
    func_env = _func_env()
    params = _make_params()
    packed = _packed_mask(func_env, _state_shopping(), params)[_head_slice("purchase")]
    assert int(packed[PURCHASE_STOP]) == 1, "STOP unselectable during shopping"


def test_no_op_masked_in_parallel_mode_as_packed():
    """No-op must stay masked on every head that has a real choice to offer.

    **Why**: ``test_no_op_masked_out_on_all_heads`` checks ``_action_masks``
    and passed throughout the misalignment.  The shift pulled each head's
    index 1 into its index 0, so the policy saw no-op as legal on heads
    where it should have been forced to act.

    Heads with nothing legal are exempt: they fall back to no-op on purpose
    (see ``test_empty_head_falls_back_to_no_op_only``).  With the default
    params ``use_domestic_sale`` is off, so nothing reads ``price_slot`` in
    parallel mode and that head is legitimately no-op only.
    """
    func_env = _func_env()
    params = _make_params()
    packed = _packed_mask(func_env, _make_state(), params)
    for head in MASK_HEAD_ORDER:
        m = packed[_head_slice(head)]
        if int(m[1:].sum()) == 0:
            continue  # inert head, no-op is its only honest option
        assert int(m[0]) == 0, f"{head} no-op not masked despite having real options"


# ══════════════════════════════════════════════════════════════════════════════
# Every head must offer at least one selectable value
# ══════════════════════════════════════════════════════════════════════════════


def _state_no_opponent_stock():
    """Nobody holds stock — the opponent head has no rival worth targeting."""
    state = _make_state()
    return state._replace(
        factory_store=jnp.zeros_like(state.factory_store),
        harbour_store=jnp.zeros_like(state.harbour_store),
    )


def _state_all_colours_owned():
    """P0 owns every colour — no colour is buyable, and it holds no goods."""
    state = _make_state()
    return state._replace(
        factory_colors=state.factory_colors.at[0].set(1),
        factory_store=jnp.zeros_like(state.factory_store),
        harbour_store=jnp.zeros_like(state.harbour_store),
    )


def _state_broke():
    """No cash — nothing is affordable from anyone."""
    state = _make_state()
    return state._replace(cash=jnp.zeros_like(state.cash))


@pytest.mark.parametrize(
    "name,builder",
    [
        ("parallel", _state_parallel),
        ("no_opponent_stock", _state_no_opponent_stock),
        ("all_colours_owned", _state_all_colours_owned),
        ("broke", _state_broke),
        ("auction_legal", _state_auction_legal),
        ("auction_active", _state_auction_active),
        ("shopping", _state_shopping),
        ("produce", _state_produce),
    ],
)
def test_every_head_has_a_selectable_value(name, builder):
    """No head may be entirely masked out.

    **Why**: sb3-contrib's ``MaskableCategorical`` does not reject an
    all-zero mask — it renormalises to a *uniform* distribution over every
    value, valid or not, with ``log_prob`` 0 and entropy 0.  So an empty
    head makes the policy emit invalid values at random on that head while
    receiving no gradient for them.  It fails silently, which is why this
    needs asserting rather than trusting.
    """
    func_env = _func_env()
    params = _make_params()
    masks = func_env._action_masks(builder(), params)
    for head in MASK_HEAD_ORDER:
        assert int(np.asarray(masks[head]).sum()) > 0, (
            f"{head} head fully masked in {name} state"
        )


@pytest.mark.parametrize(
    "name,builder",
    [
        ("no_opponent_stock", _state_no_opponent_stock),
        ("all_colours_owned", _state_all_colours_owned),
        ("broke", _state_broke),
    ],
)
def test_empty_head_falls_back_to_no_op_only(name, builder):
    """When a head has nothing legal, no-op must be its *only* option.

    **Why**: the fallback has to re-enable no-op without inventing a legal
    parameter value.  If it opened up anything else, the policy could pick
    a colour it cannot buy or an opponent with no stock and the action
    handler would silently clamp it to a real target.
    """
    func_env = _func_env()
    params = _make_params()
    masks = func_env._action_masks(builder(), params)
    for head in MASK_HEAD_ORDER:
        m = np.asarray(masks[head])
        if int(m[1:].sum()) == 0:
            assert int(m[0]) == 1, f"{head} empty in {name} but no-op not re-enabled"
            assert int(m.sum()) == 1, f"{head} fallback opened extra values in {name}"


def test_no_op_fallback_only_fires_on_inert_heads():
    """A head may only fall back to no-op when nothing reads it.

    **Why**: this is the safety argument for the fallback.  If a head were
    empty while an action_type that consumes it was still offered, the
    policy could select that action with a no-op parameter and the handler
    would clamp the no-op into a real index — a silently wrong move rather
    than a masked one.
    """
    func_env = _func_env()
    params = _make_params()
    consumers = {
        "opponent": (ACTION_BUY_FROM_FACTORY_STORE, ACTION_MOVE_LOAD),
        "color": (ACTION_BUY_FACTORY, ACTION_DOMESTIC_SALE),
        "price_slot": (ACTION_DOMESTIC_SALE,),
    }
    for name, builder in [
        ("parallel", _state_parallel),
        ("no_opponent_stock", _state_no_opponent_stock),
        ("all_colours_owned", _state_all_colours_owned),
        ("broke", _state_broke),
    ]:
        masks = func_env._action_masks(builder(), params)
        at = np.asarray(masks["action_type"])
        for head, action_types in consumers.items():
            m = np.asarray(masks[head])
            if int(m[1:].sum()) == 0:
                for a in action_types:
                    assert int(at[a + 1]) == 0, (
                        f"{name}: {head} head is inert but action_type {a} "
                        f"is still offered and would read it"
                    )
