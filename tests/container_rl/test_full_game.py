"""One complete game — a human at seat 0, two AI opponents — with every move played.

The server has plenty of tests for a single move in a hand-built position.
What it did not have is a *game*: seat 0 sending the same multi-head actions
the TUI sends, two AI seats answering through :class:`GameManager`, state
round-tripping through the database on every step, and the whole thing running
from the opening turn to ``game_over``.  Every move in the rulebook is played
by the human at least once on the way there, so this is a coverage run as much
as a game.

Reaching a move is mostly a matter of reaching the position that allows it, and
those positions form a chain.  The ship starts at sea and empty; it only enters
a harbour by loading from an opponent, only reaches the open sea again from a
harbour, and can only be auctioned from the open sea with cargo aboard.
Loading needs an opponent with stock in *their* harbour, which needs them to
have bought from somebody's factory store, which needs somebody to have
produced.  So the human below plays a *coverage-seeking* policy rather than a
good one: at every decision it takes the first move it has not played yet, and
when there is nothing new within reach it plays whatever keeps containers
moving — produce cheaply, buy storage, sell on — so the opponents' stores keep
filling and the rest of the chain stays reachable.

The AI seats have no checkpoint configured, so they take the masked-random
fallback — the same path a missing or stale model takes in production.  That
fallback and the env are both seeded, so this is the same game every run.

Two deliberate choices worth knowing about:

* The human's actions are built with the TUI's own ``_mh`` helper.  A test that
  hand-rolled the head encoding would still pass if the client and the env
  disagreed about it; borrowing the client's encoder means the bytes under test
  are the bytes a player actually sends.
* The game runs the shipped configuration, which offers ten of the env's
  eleven action types: ``ACTION_DOMESTIC_SALE`` needs ``use_domestic_sale``,
  and the server never sets it.  The eleventh therefore gets its own test —
  which currently xfails, because turning the flag on breaks mask building
  outright.  ``test_the_domestic_sale_variant_is_playable`` has the details.
* Two moves cannot be reached by playing well, only by being offered them.
  Seat 0 bids only when an opponent sells, and an opponent playing at random
  may never get a ship to sea, so that one is staged directly in
  ``test_seat_zero_can_bid_in_an_opponents_auction`` instead of being left to
  the run.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from container_rl.client.tui import _mh
from container_rl.env.container import (
    ACTION_BUY_FACTORY,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_BUY_WAREHOUSE,
    ACTION_DOMESTIC_SALE,
    ACTION_MOVE_AUCTION,
    ACTION_MOVE_LOAD,
    ACTION_MOVE_SEA,
    ACTION_PASS,
    ACTION_PRODUCE,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    FACTORY_STORAGE_MULTIPLIER,
    LEAVE_IDLE,
    LOAN_AMOUNT,
    LOCATION_HARBOUR_OFFSET,
    LOCATION_OPEN_SEA,
    MAX_EPISODE_STEPS,
    MAX_FACTORIES_PER_PLAYER,
    MAX_WAREHOUSES_PER_PLAYER,
    PRICE_SLOTS,
    PRODUCE_PRICE_CHOICES,
    PURCHASE_STOP,
    SHIP_CAPACITY,
)
from container_rl.server.database import Database
from container_rl.server.game_manager import GameManager

HUMAN = 0
NUM_PLAYERS = 3          # the tester plus two AI opponents
NUM_COLORS = 5
SEED = 7

# The env is stepped eagerly (``game_manager`` disables jit, because the
# observation cannot be traced while the player count lives in the params), so
# a step costs real time.  This is the backstop for a game that somehow never
# reaches ``game_over``: the assertion is on ``game_over`` itself, not on this.
MAX_DECISIONS = 2000

_MOVE_NAMES = {
    ACTION_BUY_FACTORY: "buy factory",
    ACTION_BUY_WAREHOUSE: "buy warehouse",
    ACTION_PRODUCE: "produce",
    ACTION_BUY_FROM_FACTORY_STORE: "buy from a factory store",
    ACTION_MOVE_LOAD: "load the ship",
    ACTION_MOVE_SEA: "move to the open sea",
    ACTION_MOVE_AUCTION: "hold an auction",
    ACTION_PASS: "pass",
    ACTION_TAKE_LOAN: "take a loan",
    ACTION_REPAY_LOAN: "repay a loan",
    ACTION_DOMESTIC_SALE: "domestic sale",
}

# The ten moves the shipped game offers.  Ordered by dependency: a loan has to
# exist before it can be repaid, and the ship has to be loaded before it can
# put to sea and before there is anything to auction.
SHIPPED_MOVES = (
    ACTION_PASS,
    ACTION_BUY_FACTORY,
    ACTION_PRODUCE,
    ACTION_BUY_WAREHOUSE,
    ACTION_TAKE_LOAN,
    ACTION_REPAY_LOAN,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_MOVE_LOAD,
    ACTION_MOVE_SEA,
    ACTION_MOVE_AUCTION,
)

# What the human does when it has nothing new to reach for.  Finishing the
# shipping chain comes first (a loaded ship blocks nothing else, but a ship
# stuck in a harbour cannot be reloaded), then producing, then buying the
# factories and warehouses that let it produce again.
#
# That last part is not padding.  A player's factory store holds two containers
# per factory and only empties when an *opponent* buys out of it, so a seat with
# one factory produces twice and then cannot produce again until somebody shops
# there.  Buying factories is what keeps the produce menu reachable often enough
# to play all of it.
FALLBACK_MOVES = (
    ACTION_MOVE_AUCTION,
    ACTION_MOVE_SEA,
    ACTION_MOVE_LOAD,
    ACTION_PRODUCE,
    ACTION_BUY_FACTORY,
    ACTION_BUY_WAREHOUSE,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_PASS,
)


class HumanSeat:
    """The human at seat 0, playing for coverage rather than for points.

    Named ``HumanSeat`` rather than ``Tester`` only because pytest tries to
    collect anything starting with "Test".

    Tracks two things: ``moves`` — the action types it has played — and
    ``moments`` — the sub-decisions that are moves in their own right as far as
    a player is concerned (a bid, a seller's accept and reject, leaving a
    factory idle, walking away from a shop) but that share an action type with
    something else.
    """

    def __init__(self, num_players: int, num_colors: int, use_domestic_sale: bool):
        self.num_players = num_players
        self.num_colors = num_colors
        self.wanted = list(SHIPPED_MOVES)
        if use_domestic_sale:
            self.wanted.append(ACTION_DOMESTIC_SALE)
        self.moves: set[int] = set()
        self.moments: set[str] = set()
        self._produce_price = 0      # how far through the produce menu it is
        self._shop_purchases = 0     # purchases made in the current shopping run

    # -- reading the position ------------------------------------------

    def _cash(self, st) -> int:
        return int(st.cash[HUMAN])

    def _cargo(self, st) -> int:
        return int(np.sum(np.asarray(st.ship_contents[HUMAN]) > 0))

    def _affordable_opponent(self, st, store) -> int | None:
        """Relative index of an opponent this seat can afford to buy from.

        The opponent head counts clockwise from the player's left, matching
        ``_get_target_player``: index 0 is the next seat round, and index 0 on
        the mask is the no-op the TUI's ``_mh`` shifts past.
        """
        cash = self._cash(st)
        for i in range(self.num_players - 1):
            rows = np.asarray(store[(HUMAN + 1 + i) % self.num_players])
            for slot in range(PRICE_SLOTS):
                if slot + 1 <= cash and rows[:, slot].any():
                    return i
        return None

    def _own_container(self, st) -> tuple[int, int] | None:
        """A (colour, price slot) this seat actually holds, for a domestic sale.

        The colour and price-slot heads are masked independently, so a pair
        picked off the masks alone can name a slot the player has nothing in —
        and the sale then silently does nothing.  Read the stores instead.
        """
        factory = np.asarray(st.factory_store[HUMAN])
        harbour = np.asarray(st.harbour_store[HUMAN])
        for colour in range(self.num_colors):
            for slot in range(PRICE_SLOTS):
                if factory[colour, slot] > 0 or harbour[colour, slot] > 0:
                    return colour, slot
        return None

    # -- building one move ---------------------------------------------

    def build(self, st, move: int) -> list[int] | None:
        """The action for *move* in this position, or None if it is not on."""
        cash = self._cash(st)

        if move == ACTION_PASS:
            return _mh(move)

        if move == ACTION_BUY_FACTORY:
            owned = np.asarray(st.factory_colors[HUMAN])
            if owned.sum() >= MAX_FACTORIES_PER_PLAYER:
                return None
            if cash < (int(owned.sum()) + 1) * 3:
                return None
            free = [c for c in range(self.num_colors) if owned[c] == 0]
            return _mh(move, color=free[0]) if free else None

        if move == ACTION_BUY_WAREHOUSE:
            held = int(st.warehouse_count[HUMAN])
            if held >= MAX_WAREHOUSES_PER_PLAYER or cash < held + 3:
                return None
            return _mh(move)

        if move == ACTION_PRODUCE:
            if int(st.produced_this_turn) > 0 or cash < 1:
                return None
            owned = np.asarray(st.factory_colors[HUMAN])
            stored = int(np.asarray(st.factory_store[HUMAN]).sum())
            if stored >= int(owned.sum()) * FACTORY_STORAGE_MULTIPLIER:
                return None
            supply = np.asarray(st.container_supply)
            if not any(owned[c] and supply[c] > 0 for c in range(self.num_colors)):
                return None
            return _mh(move)

        if move == ACTION_BUY_FROM_FACTORY_STORE:
            if int(np.asarray(st.harbour_store[HUMAN]).sum()) >= int(st.warehouse_count[HUMAN]):
                return None  # nowhere to put it
            opp = self._affordable_opponent(st, st.factory_store)
            return _mh(move, opp=opp) if opp is not None else None

        if move == ACTION_MOVE_LOAD:
            if self._cargo(st) >= SHIP_CAPACITY:
                return None
            opp = self._affordable_opponent(st, st.harbour_store)
            return _mh(move, opp=opp) if opp is not None else None

        if move == ACTION_MOVE_SEA:
            in_harbour = int(st.ship_location[HUMAN]) >= LOCATION_HARBOUR_OFFSET
            return _mh(move) if in_harbour else None

        if move == ACTION_MOVE_AUCTION:
            at_sea = int(st.ship_location[HUMAN]) == LOCATION_OPEN_SEA
            return _mh(move) if at_sea and self._cargo(st) > 0 else None

        if move == ACTION_TAKE_LOAN:
            return _mh(move) if int(st.loans[HUMAN]) < 2 else None

        if move == ACTION_REPAY_LOAN:
            if int(st.loans[HUMAN]) < 1 or cash < LOAN_AMOUNT:
                return None
            return _mh(move)

        if move == ACTION_DOMESTIC_SALE:
            held = self._own_container(st)
            return _mh(move, color=held[0], slot=held[1]) if held else None

        raise AssertionError(f"unknown move {move}")

    # -- one decision ---------------------------------------------------

    def decide(self, st) -> list[int]:
        """What this seat plays in *st*.  Never None: passing is always legal."""
        if int(st.auction_active) > 0:
            return self._decide_auction(st)
        if int(st.produce_active) > 0:
            return self._decide_produce(st)
        if int(st.shopping_active) > 0:
            return self._decide_shopping(st)

        self._shop_purchases = 0
        for move in self.wanted:
            if move in self.moves:
                continue
            action = self.build(st, move)
            if action is not None:
                self.moves.add(move)
                return action
        for move in FALLBACK_MOVES:
            action = self.build(st, move)
            if action is not None:
                self.moves.add(move)
                return action
        self.moves.add(ACTION_PASS)
        return _mh(ACTION_PASS)

    def _decide_auction(self, st) -> list[int]:
        """Bid, or — as the seller — take the highest bid or keep the cargo.

        Both sides use the raw ``[auction, bidder, …, amount]`` array the TUI
        sends; the bidder head has to name this seat or the server refuses it.
        """
        if int(st.auction_seller) != HUMAN:
            bid = min(self._cash(st), 2)
            self.moments.add("bid" if bid > 0 else "bid nothing")
            return [ACTION_MOVE_AUCTION + 1, HUMAN, 0, 0, bid]
        # Seller's decision.  Reject the first auction and accept the next, so
        # both halves of the branch are played.
        if "reject a bid" not in self.moments:
            self.moments.add("reject a bid")
            return [ACTION_MOVE_AUCTION + 1, HUMAN, 0, 0, 0]
        self.moments.add("accept a bid")
        return [ACTION_MOVE_AUCTION + 1, HUMAN, 0, 0, 1]

    def _decide_produce(self, st) -> list[int]:
        """Price one pending factory, or leave it idle.

        Chances to produce are scarcer than they look — see ``FALLBACK_MOVES``
        — so the menu is played out in the order the choices matter: fill a
        container, then leave one idle, then a second price, and everything
        after that at $1.  Cheap containers are the ones opponents buy, and
        containers changing hands is what empties the store and buys the next
        chance to produce.
        """
        pending = np.flatnonzero(np.asarray(st.produce_pending))
        colour = int(pending[0]) if pending.size else 0
        opening = (0, LEAVE_IDLE, 1)
        slot = opening[self._produce_price] if self._produce_price < len(opening) else 0
        self._produce_price = min(self._produce_price + 1, len(opening))
        assert slot == LEAVE_IDLE or slot < PRODUCE_PRICE_CHOICES
        self.moments.add(
            "leave a factory idle" if slot == LEAVE_IDLE else f"produce at ${slot + 1}"
        )
        return _mh(ACTION_PRODUCE, color=colour, slot=slot)

    def _decide_shopping(self, st) -> list[int]:
        """Buy one container, or walk away — the two things the menu offers.

        Walking away has to be done deliberately.  The env ends a shopping run
        by itself the moment nothing affordable is left, so a shopper that only
        ever stops when it has run out never plays the STOP the menu shows it.
        A run is only ever entered when at least one purchase is possible,
        though, so stopping on the *first* prompt is always available — and
        that is what this does, once, after a purchase of the same kind is
        already on the record so nothing is starved by it.
        """
        shop = int(st.shopping_action_type)
        bought = ("buy into the harbour" if shop == ACTION_BUY_FROM_FACTORY_STORE
                  else "load a container")
        walk_away = bought in self.moments and "stop shopping" not in self.moments

        colour = None
        if not walk_away and self._shop_purchases == 0:
            target = int(st.shopping_target)
            store = (st.factory_store if shop == ACTION_BUY_FROM_FACTORY_STORE
                     else st.harbour_store)
            rows = np.asarray(store[target])
            cash = self._cash(st)
            for c in range(self.num_colors):
                if any(rows[c, s] > 0 and s + 1 <= cash for s in range(PRICE_SLOTS)):
                    colour = c
                    break

        if colour is None:
            self._shop_purchases = 0
            self.moments.add("stop shopping")
            return _mh(shop, purchase=PURCHASE_STOP)

        self._shop_purchases += 1
        self.moments.add(bought)
        # Purchase index 1 is $2 for a factory buy (the cheapest harbour price
        # on offer) and the plain "buy this one" signal for a ship load.
        return _mh(shop, color=colour, purchase=1)


def _human_to_move(st) -> bool:
    """Is the game waiting on seat 0?

    An auction is the one move made out of turn — the seller stays
    ``current_player`` while everybody else answers — so whose turn it is does
    not settle the question on its own.
    """
    if int(st.game_over) > 0:
        return False
    if int(st.auction_active) > 0:
        if int(st.auction_seller) != HUMAN:
            return int(st.auction_bids[HUMAN]) < 0    # -1 means "has not bid"
        return int(st.auction_round) == 1             # the seller's decision
    return int(st.current_player) == HUMAN


def play_full_game(db_path, *, use_domestic_sale=False, num_colors=NUM_COLORS, seed=SEED):
    """Play one game to ``game_over`` and return (manager, game_id, tester, sent)."""
    np.random.seed(seed)                 # the AI's masked-random fallback
    sent: list[tuple[str, dict]] = []
    manager = GameManager(Database(str(db_path)),
                          lambda gid, event, payload: sent.append((event, payload)))

    game = manager.create_game_trusted(
        "tester", num_players=NUM_PLAYERS, num_colors=num_colors,
        seed=seed, ai_count=NUM_PLAYERS - 1,
    )
    game_id = game["game_id"]
    assert game["player_index"] == HUMAN
    assert manager.maybe_start_game(game_id), "the AI seats did not fill the lobby"

    env = manager.load_or_create_env(game_id)
    if use_domestic_sale:
        # The server never turns the variant on, so the only way to play the
        # move is to enable it on the live env once the game exists.
        env.func_env.params = env.func_env.params.replace(use_domestic_sale=True)
    manager.play_ai_turn_if_needed(game_id)

    tester = HumanSeat(NUM_PLAYERS, num_colors, use_domestic_sale)
    stalled = 0
    for _ in range(MAX_DECISIONS):
        state = manager.get_state(game_id)
        if int(state.game_over) > 0:
            break

        if not _human_to_move(state):
            before = int(state.step_count)
            manager.play_ai_turn_if_needed(game_id)
            stalled = stalled + 1 if int(manager.get_state(game_id).step_count) == before else 0
            assert stalled < 3, (
                "the game is waiting on nobody: it is not seat 0's move and the "
                f"AI seats played nothing (current_player="
                f"{int(state.current_player)}, auction_active="
                f"{int(state.auction_active)}, round={int(state.auction_round)})"
            )
            continue

        stalled = 0
        result = manager.process_action(game_id, HUMAN, tester.decide(state))
        assert not result.get("error"), f"the server refused seat 0's move: {result['desc']}"

    return manager, game_id, tester, sent


def _missing(played: set[int], wanted) -> list[str]:
    return [_MOVE_NAMES[m] for m in wanted if m not in played]


@pytest.fixture(scope="module")
def finished_game(tmp_path_factory):
    """One game, played once, shared by every assertion below.

    The env is stepped eagerly — ``game_manager`` disables jit because the
    observation cannot be traced while the player count lives in the params —
    so a whole game costs minutes, not milliseconds.  Playing it once per
    module and asserting against the result keeps the cost to one game while
    still letting each thing it pins fail under its own name.
    """
    return play_full_game(tmp_path_factory.mktemp("full-game") / "game.db")


@pytest.mark.slow
def test_the_game_runs_from_the_opening_turn_to_the_end(finished_game):
    """Seat 0 and the two AI seats between them get the game to ``game_over``.

    ``_check_game_end`` knows two endings: two colours run out of containers,
    or the episode hits ``MAX_EPISODE_STEPS``.  Opponents playing at random
    waste too many actions to drain two colours in a thousand steps, so this
    game reliably ends on the step cap — which is a real ending, and the one
    the comment in ``_check_game_end`` says is there to catch exactly this.
    Pinning both means ``game_over`` set by some third route is a failure.
    """
    manager, game_id, _tester, _sent = finished_game
    state = manager.get_state(game_id)
    supply = np.asarray(state.container_supply)

    assert int(state.game_over) > 0, (
        f"the game never ended (step {int(state.step_count)}); supply {supply.tolist()}"
    )
    assert int((supply <= 0).sum()) >= 2 or int(state.step_count) > MAX_EPISODE_STEPS, (
        f"the game ended for no reason the rules give: step {int(state.step_count)}, "
        f"supply {supply.tolist()}"
    )


@pytest.mark.slow
def test_every_shipped_move_is_played_at_least_once(finished_game):
    """The point of the run: seat 0 plays every move the game offers."""
    _manager, _game_id, tester, _sent = finished_game

    missing = _missing(tester.moves, SHIPPED_MOVES)
    assert not missing, "seat 0 never played: " + ", ".join(missing)


@pytest.mark.slow
def test_the_seller_answers_its_own_auctions_both_ways(finished_game):
    """Taking the money and keeping the cargo are two moves, not one.

    Only the seller's half is pinned here.  Bidding is the other half, and
    seat 0 is only ever asked to bid if an *opponent* opens an auction — which
    needs an AI to load a ship and reach the open sea, and one playing at
    random may never do it.  ``test_seat_zero_can_bid_in_an_opponents_auction``
    stages that position directly rather than hoping the game produces it.
    """
    _manager, _game_id, tester, _sent = finished_game

    assert "accept a bid" in tester.moments, "seat 0 never took a bid"
    assert "reject a bid" in tester.moments, "seat 0 never turned a bid down"


@pytest.mark.slow
def test_the_produce_and_shopping_menus_are_played_out(finished_game):
    """Both branches of each continuation menu a player is shown on screen.

    A continuation is not a separate action type, so none of this shows up in
    the move coverage above — but filling a factory and leaving it idle are
    different answers to the same prompt, and so are buying a container and
    walking away from the shop.
    """
    _manager, _game_id, tester, _sent = finished_game

    for moment in ("produce at $1", "leave a factory idle",
                   "buy into the harbour", "load a container", "stop shopping"):
        assert moment in tester.moments, (
            f"seat 0 never got to {moment}; it played: {sorted(tester.moments)}"
        )


@pytest.mark.slow
def test_the_table_is_told_about_the_finish(finished_game):
    """Clients only redraw on a broadcast, so the end has to be sent."""
    _manager, _game_id, _tester, sent = finished_game

    updates = [payload for event, payload in sent if event == "state_update"]
    assert updates, "nothing was ever broadcast"
    assert int(updates[-1]["game_over"]) > 0, "the last thing sent was not the finish"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="a game that ends on an AI's move can leave the row 'active' — see docstring",
    strict=False,
)
def test_a_finished_game_is_marked_finished(finished_game):
    """The row this game leaves behind says it is still being played.

    ``_play_ai_turns`` only writes ``finished`` on the branch it takes after
    stepping an AI seat.  Three of its four exits — the game already over at
    the top of the loop, an auction still owed a human answer, and the next
    turn belonging to a human — leave the loop without writing anything, and
    the env can cross the end of the game inside ``_play_ai_auction_bids``
    just before one of them.  ``process_action`` cannot cover for it either:
    it refuses outright once ``game_over`` is set, so no later move puts the
    status right.

    A row left on ``active`` is not cosmetic.  ``list_joinable_games`` shows
    active games back to the players who were in them, so the finished game
    keeps being offered as one to rejoin, and ``finished_at`` is never
    stamped.  Marked xfail rather than deleted: the assertion is what the
    server should do, and this is the test that should go green when it does.
    """
    manager, game_id, _tester, _sent = finished_game

    assert manager.db.get_game_by_id(game_id)["status"] == "finished"


def _staged_game(tmp_path):
    """A started game with seat 0 human and seats 1-2 AI, nothing played yet.

    Returns ``(manager, game_id, env, sent)``; the caller puts the pieces where
    it needs them and drives from there.
    """
    sent: list[tuple[str, dict]] = []
    manager = GameManager(Database(str(tmp_path / "game.db")),
                          lambda gid, event, payload: sent.append((event, payload)))
    game = manager.create_game_trusted(
        "tester", num_players=NUM_PLAYERS, num_colors=NUM_COLORS,
        seed=SEED, ai_count=NUM_PLAYERS - 1,
    )
    game_id = game["game_id"]
    assert manager.maybe_start_game(game_id)
    env = manager.load_or_create_env(game_id)
    return manager, game_id, env, sent


def test_seat_zero_can_bid_in_an_opponents_auction(tmp_path):
    """The one move a player makes out of turn, staged rather than waited for.

    Seat 0 is only asked to bid when somebody else sells, so the full game
    cannot be relied on to produce it: an AI playing at random has to load a
    ship and put to sea first, and it may never get there.  Putting an AI seat
    at sea with cargo and letting it sell reaches the same prompt in one move.

    The bid has to name seat 0 on the bidder head or the server refuses it —
    that head is a raw player index, so it is the only thing stopping one seat
    bidding in another's name.
    """
    np.random.seed(SEED)
    seller = 1
    manager, game_id, env, _sent = _staged_game(tmp_path)
    base = env.state
    env.state = base._replace(
        current_player=jnp.asarray(seller, dtype=base.current_player.dtype),
        ship_contents=base.ship_contents.at[seller].set(
            jnp.asarray([1, 2, 0, 0, 0], dtype=base.ship_contents.dtype)),
        ship_location=base.ship_location.at[seller].set(
            jnp.asarray(LOCATION_OPEN_SEA, dtype=base.ship_location.dtype)),
    )

    manager.process_action(game_id, seller, _mh(ACTION_MOVE_AUCTION))

    state = manager.get_state(game_id)
    assert int(state.auction_active) == 1, "the AI's auction closed without seat 0"
    assert int(state.auction_bids[HUMAN]) < 0, "seat 0's bid was entered for it"

    tester = HumanSeat(NUM_PLAYERS, NUM_COLORS, use_domestic_sale=False)
    result = manager.process_action(game_id, HUMAN, tester.decide(state))

    assert not result.get("error"), f"the server refused seat 0's bid: {result['desc']}"
    assert {"bid", "bid nothing"} & tester.moments, "seat 0 did not play a bid"
    # Every bid is in now, so the auction has either moved to the seller's
    # decision or resolved outright.  Either way seat 0's answer counted.
    after = manager.get_state(game_id)
    assert int(after.auction_active) == 0 or int(after.auction_round) == 1


@pytest.mark.xfail(
    reason="use_domestic_sale=True cannot build its masks at all — see docstring",
    strict=True,
)
def test_the_domestic_sale_variant_is_playable(tmp_path):
    """The eleventh move cannot be played, because the variant does not run.

    ``ACTION_DOMESTIC_SALE`` only appears on the menu when
    ``use_domestic_sale`` is set, and the server never sets it — so the full
    game above covers the ten moves the shipped configuration offers and this
    is the eleventh, tested on its own.

    It does not get as far as the move.  The price-slot mask that the flag
    switches on reads a whole price row, ``state.factory_store[player, c]``,
    which is ``PRICE_SLOTS`` wide, and tries to ``jnp.where`` it against a mask
    that is ``PRICE_SLOTS + 1`` wide because of the no-op at index 0.  Those do
    not broadcast, so ``_action_masks`` raises for every state — and since
    ``observation`` builds the masks, an env with the variant on cannot even
    ``reset``.  Strict xfail: when the widths are reconciled this should start
    passing, and it should be noticed when it does.
    """
    manager, game_id, env, sent = _staged_game(tmp_path)
    env.func_env.params = env.func_env.params.replace(use_domestic_sale=True)

    state = manager.get_state(game_id)
    assert int(state.current_player) == HUMAN

    # Reachable, not merely accepted: the env dispatches the action whatever
    # the flag says, so what the flag has to buy is a legal move on the menu.
    masks = env.func_env._action_masks(state, env.func_env.params)
    assert int(masks["action_type"][ACTION_DOMESTIC_SALE + 1]) == 1, \
        "the variant is on but the move is still masked out"

    tester = HumanSeat(NUM_PLAYERS, NUM_COLORS, use_domestic_sale=True)
    action = tester.build(state, ACTION_DOMESTIC_SALE)
    assert action is not None, "seat 0 starts with a container; it has one to sell"

    cash_before = int(state.cash[HUMAN])
    stock_before = int(np.asarray(state.factory_store[HUMAN]).sum())

    sent.clear()
    result = manager.process_action(game_id, HUMAN, action)
    assert not result.get("error"), result["desc"]

    # Read the sale off the first broadcast rather than the live state: the
    # same call plays the AI seats on, and an opponent buying from seat 0's
    # store would move its cash again before the assertion got there.
    after = next(payload["state_data"] for event, payload in sent if event == "state_update")
    assert after["cash"][HUMAN] == cash_before + 2, "the sale paid nothing"
    assert int(np.asarray(after["factory_store"][HUMAN]).sum()) == stock_before - 1, \
        "the container was sold but never left the store"
