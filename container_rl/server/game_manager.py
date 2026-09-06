"""Game lifecycle manager — create, join, start, and orchestrate turns."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
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
    HEAD_OPPONENT,
    HEAD_PURCHASE,
    LEAVE_IDLE,
    PURCHASE_STOP,
    ActionEncoder,
    ContainerJaxEnv,
    EnvState,
    mask_size,
)
from container_rl.server.database import Database
from container_rl.server.protocol import deserialize_state, serialize_state

jax.config.update("jax_disable_jit", True)

_ACTION_NAMES = {
    ACTION_BUY_FACTORY: "Buy factory",
    ACTION_BUY_WAREHOUSE: "Buy warehouse",
    ACTION_PRODUCE: "Produce",
    ACTION_BUY_FROM_FACTORY_STORE: "Buy from factory",
    ACTION_MOVE_LOAD: "Load ship",
    ACTION_MOVE_SEA: "Move to sea",
    ACTION_MOVE_AUCTION: "Auction",
    ACTION_PASS: "Pass",
    ACTION_TAKE_LOAN: "Take loan",
    ACTION_REPAY_LOAN: "Repay loan",
    ACTION_DOMESTIC_SALE: "Domestic sale",
}

logger = logging.getLogger(__name__)

# Head order the MultiDiscrete policy expects; must match ``head_sizes``.
_MASK_HEADS = ("action_type", "opponent", "color", "price_slot", "purchase")

# Backstops so a misbehaving model cannot spin forever holding the manager lock.
MAX_AI_TURN_STEPS = 200
MAX_AI_CONTINUATION_STEPS = 50


class GameManager:
    def __init__(self, db: Database, broadcast: Callable[[int, str, Any], None],
                 ai_model_path: str | None = None):
        self.db = db
        self._broadcast = broadcast
        self._envs: dict[int, ContainerJaxEnv] = {}
        self._encoders: dict[int, ActionEncoder] = {}
        self._lock = threading.RLock()
        self._ai_model_path = ai_model_path
        self._ai_model: Any = None  # MaskablePPO instance (lazy-loaded)
        self._ai_model_failed = False  # load/inference failed; use random legal play
        self._ai_slots: dict[int, list[int]] = {}  # game_id -> [ai_player_indices]
        self._mask_sizes: dict[tuple[int, int], int] = {}  # cache mask_size per (np, nc)

    # ------------------------------------------------------------------
    # create / join
    # ------------------------------------------------------------------

    def create_game(
        self, player_name: str, password: str | None,
        num_players: int, num_colors: int = 5, seed: int | None = None,
        ai_count: int = 0,
    ) -> dict:
        """Create a new game and join as the first player (slot 0)."""
        player_id = self.db.upsert_player(player_name, password)
        game_id, code = self.db.create_game(num_players, num_colors, seed)
        self.db.assign_player_slot(game_id, player_id, 0)
        self._seed_ai_slots(game_id, num_players, ai_count)
        return {"game_id": game_id, "code": code, "player_index": 0, "ai_count": ai_count}

    def create_game_trusted(
        self, player_name: str,
        num_players: int, num_colors: int = 5, seed: int | None = None,
        ai_count: int = 0,
    ) -> dict:
        """Create a new game with a trusted player name (no password check).

        *ai_count*: how many AI opponents (filled from the last slots).
        """
        player_id = self.db.upsert_player_trusted(player_name)
        game_id, code = self.db.create_game(num_players, num_colors, seed)
        self.db.assign_player_slot(game_id, player_id, 0)
        self._seed_ai_slots(game_id, num_players, ai_count)
        return {"game_id": game_id, "code": code, "player_index": 0, "ai_count": ai_count}

    def _seed_ai_slots(self, game_id: int, num_players: int, ai_count: int) -> None:
        """Pre-assign the trailing slots to AI players so the lobby fills."""
        if ai_count <= 0:
            return
        ai_slots = list(range(num_players - ai_count, num_players))
        self._ai_slots[game_id] = ai_slots
        for slot in ai_slots:
            ai_pid = self.db.upsert_player_trusted(f"[AI] Player {slot + 1}")
            self.db.assign_player_slot(game_id, ai_pid, slot, is_ai=True)

    def _get_ai_slots(self, game_id: int) -> list[int]:
        """Derive AI player slots from the database (survives server restarts).

        Reads the ``is_ai`` flag on the slot rather than sniffing the display
        name, so a human calling themselves "[AI] …" is not auto-played.
        """
        if game_id in self._ai_slots:
            return self._ai_slots[game_id]
        players = self.db.get_game_players(game_id)
        ai_slots = [int(p["player_index"]) for p in players if p.get("is_ai")]
        if ai_slots:
            self._ai_slots[game_id] = ai_slots
        return ai_slots

    def join_game(
        self, player_name: str, password: str | None, code: str,
    ) -> dict:
        """Join a lobby game or reconnect to an active game."""
        game = self.db.get_game_by_code(code)
        if not game:
            raise ValueError(f"Game '{code}' not found.")
        if game["status"] not in ("lobby", "active"):
            raise ValueError(f"Game '{code}' is not accepting players (status: {game['status']}).")
        player_id = self.db.upsert_player(player_name, password)

        if game["status"] == "active":
            player_index = self.db.get_player_slot(game["id"], player_id)
            if player_index is None:
                raise ValueError(f"Player '{player_name}' is not in game '{code}'.")
        else:
            player_index = self.db.assign_player_slot(game_id=game["id"], player_id=player_id)

        return {
            "game_id": game["id"],
            "code": code,
            "player_index": player_index,
            "num_players": game["num_players"],
            "num_colors": game["num_colors"],
            "status": game["status"],
        }

    def join_game_trusted(
        self, player_name: str, code: str,
    ) -> dict:
        """Join a game with a trusted player name (no password check)."""
        game = self.db.get_game_by_code(code)
        if not game:
            raise ValueError(f"Game '{code}' not found.")
        if game["status"] not in ("lobby", "active"):
            raise ValueError(f"Game '{code}' is not accepting players (status: {game['status']}).")
        player_id = self.db.upsert_player_trusted(player_name)

        if game["status"] == "active":
            player_index = self.db.get_player_slot(game["id"], player_id)
            if player_index is None:
                raise ValueError(f"Player '{player_name}' is not in game '{code}'.")
        else:
            player_index = self.db.assign_player_slot(game_id=game["id"], player_id=player_id)

        return {
            "game_id": game["id"],
            "code": code,
            "player_index": player_index,
            "num_players": game["num_players"],
            "num_colors": game["num_colors"],
            "status": game["status"],
        }

    def list_joinable(self, player_name: str | None = None) -> list[dict]:
        return self.db.list_joinable_games(player_name)

    def get_game_players(self, game_id: int) -> list[dict]:
        return self.db.get_game_players(game_id)

    def get_game_info(self, game_id: int) -> dict:
        game = self.db.get_game_by_id(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found.")
        return game

    # ------------------------------------------------------------------
    # start / resume
    # ------------------------------------------------------------------

    def maybe_start_game(self, game_id: int) -> bool:
        """Check if the lobby is full; if so, create the env and save initial state."""
        with self._lock:
            game = self.db.get_game_by_id(game_id)
            if game is None:
                return False
            if game["status"] != "lobby":
                return game["status"] == "active"
            filled = self.db.count_game_players(game_id)
            if filled < game["num_players"]:
                return False

            env = ContainerJaxEnv(
                num_players=game["num_players"],
                num_colors=game["num_colors"],
            )
            env.reset(seed=game["seed"])
            self._envs[game_id] = env
            self._encoders[game_id] = ActionEncoder(game["num_players"], game["num_colors"])
            self._save_state(game_id, env.state)
            self.db.set_game_status(game_id, "active")
            # Deliberately does *not* auto-play AI turns here: doing so would
            # broadcast state_update before the caller has sent game_started.
            # Callers invoke play_ai_turn_if_needed() after announcing the start.
            return True

    def load_or_create_env(self, game_id: int) -> ContainerJaxEnv:
        """Return the live env for *game_id*, loading from DB if needed."""
        with self._lock:
            if game_id in self._envs:
                return self._envs[game_id]
            game = self.db.get_game_by_id(game_id)
            if game is None:
                raise ValueError(f"Game {game_id} not found.")
            env = ContainerJaxEnv(
                num_players=game["num_players"],
                num_colors=game["num_colors"],
            )
            blob = self.db.load_state(game_id)
            if blob:
                env.state = deserialize_state(blob)
            else:
                env.reset(seed=game["seed"])
            self._envs[game_id] = env
            self._encoders[game_id] = ActionEncoder(game["num_players"], game["num_colors"])
            return env

    def get_encoder(self, game_id: int) -> ActionEncoder:
        with self._lock:
            if game_id not in self._encoders:
                game = self.db.get_game_by_id(game_id)
                self._encoders[game_id] = ActionEncoder(game["num_players"], game["num_colors"])
            return self._encoders[game_id]

    def get_state(self, game_id: int) -> EnvState:
        env = self.load_or_create_env(game_id)
        return env.state

    # ------------------------------------------------------------------
    # action handling
    # ------------------------------------------------------------------

    def process_action(self, game_id: int, player_index: int, action: int | Any) -> dict:
        """Execute an action for *player_index* and broadcast the new state.

        *action* can be a flat integer (from ``ActionEncoder``) or a
        multi-head JAX array of shape ``(5,)``.
        """
        with self._lock:
            env = self.load_or_create_env(game_id)
            state = env.state
            if int(state.game_over) > 0:
                return {"turn_ended": True, "desc": "Game is over", "reward": 0.0, "game_over": True}

            actual = int(state.current_player)
            auction_active = int(state.auction_active) > 0
            # During an auction, any non-seller player can submit a bid
            # regardless of whose turn it nominally is.
            if actual != player_index and not (
                auction_active and player_index != int(state.auction_seller)
            ):
                return {
                    "turn_ended": False,
                    "desc": f"Not your turn (current player is {actual}).",
                    "reward": 0.0,
                    "game_over": False,
                    "error": True,
                }

            # An auction names its bidder by a raw player index on the
            # opponent head, and the env takes that at face value.  Left
            # unchecked, one seat can enter a bid in another's name and spend
            # their cash -- so a bid is only ever accepted for the sender.
            if auction_active and getattr(action, "ndim", 1) == 1 \
                    and hasattr(action, "__len__") and len(action) >= 5 \
                    and int(action[HEAD_OPPONENT]) != player_index:
                return {
                    "turn_ended": False,
                    "desc": "You can only bid for yourself.",
                    "reward": 0.0,
                    "game_over": False,
                    "error": True,
                }

            encoder = self.get_encoder(game_id)
            obs, reward, term, trunc, info = env.step(action)
            self._save_state(game_id, env.state)

            nc = encoder.num_colors
            db_players = self.db.get_game_players(game_id)
            pnames = {int(pl["player_index"]): pl["name"] for pl in db_players}
            # Name opponents relative to whoever acted, not to whoever is up
            # next: the turn may already have advanced by now.
            desc = _describe_action(action, encoder, nc, env.state, pnames, actor=actual)

            new_state = env.state
            turn_ended = int(new_state.current_player) != actual
            game_over = bool(term) or int(new_state.game_over) > 0

            if game_over:
                self.db.set_game_status(game_id, "finished")

            # Broadcast to all connected players for this game
            state_blob = serialize_state(new_state)
            try:
                state_data = _state_to_json_data(new_state)
            except Exception:
                state_data = {}
            self._broadcast(game_id, "state_update", {
                "state": state_blob.hex(),
                "state_data": state_data,
                "current_player": int(new_state.current_player),
                "actions_taken": int(new_state.actions_taken),
                "auction_active": int(new_state.auction_active),
                "produce_active": int(new_state.produce_active),
                "shopping_active": int(new_state.shopping_active),
                "game_over": int(new_state.game_over),
            })

            if turn_ended and not game_over:
                self._broadcast(game_id, "your_turn", {
                    "player_index": int(new_state.current_player),
                })

            # During auction, notify the bidder
            if int(new_state.auction_active) > 0:
                self._broadcast(game_id, "your_turn", {
                    "player_index": int(new_state.current_player),
                })

            # Auto-play AI turns before returning control
            if self._get_ai_slots(game_id) and not game_over:
                self._play_ai_turns(game_id)

            return {
                "turn_ended": turn_ended,
                "desc": desc,
                "reward": float(reward),
                "game_over": game_over,
            }

    # ------------------------------------------------------------------
    # AI auto-play
    # ------------------------------------------------------------------

    def _load_ai_model(self) -> Any:
        """Lazy-load the MaskablePPO model for AI opponents.

        Returns ``None`` if no model is configured or the model fails to
        load (wrong path, or a checkpoint trained against an older
        observation space).  Callers fall back to masked-random play so a
        game is never left wedged on an AI's turn.
        """
        if self._ai_model is not None:
            return self._ai_model
        if not self._ai_model_path or self._ai_model_failed:
            return None
        try:
            from sb3_contrib import MaskablePPO
            self._ai_model = MaskablePPO.load(self._ai_model_path, device="cpu")
        except Exception:
            self._ai_model_failed = True
            logger.exception(
                "Failed to load AI model from %s — AI players will play randomly.",
                self._ai_model_path,
            )
            return None
        return self._ai_model

    def _get_mask_size(self, np_: int, nc: int) -> int:
        """Return and cache the mask tail size for given player/colour counts."""
        key = (np_, nc)
        if key not in self._mask_sizes:
            self._mask_sizes[key] = mask_size(np_, nc)
        return self._mask_sizes[key]

    def _ai_step(self, env: ContainerJaxEnv, model: Any, ms_size: int) -> Any:
        """Choose one action for the current player and apply it.

        Returns whatever ``env.step`` returns.
        """
        return env.step(self._ai_choose(env, env.state, model, ms_size))

    def _ai_choose(self, env: ContainerJaxEnv, state: EnvState, model: Any,
                   ms_size: int) -> Any:
        """Pick one multi-head action for *state*'s current player.

        The state is passed in rather than read off the env because an auction
        bid is asked from a seat other than the one whose turn it is.  The
        model is advisory: if it is missing or raises (e.g. a checkpoint whose
        observation space no longer matches the env), we fall back to a uniform
        choice among the legal actions so the game keeps moving.
        """
        params = env.func_env.params
        obs_full = np.asarray(
            env.func_env.observation(state, jax.random.PRNGKey(0), params),
            dtype=np.float32,
        )
        obs = obs_full[:-ms_size] if ms_size > 0 else obs_full

        per_head = [
            np.asarray(mask, dtype=bool).reshape(-1)
            for mask in self._head_masks(env, state)
        ]
        # sb3-contrib does ``th.as_tensor(masks).view(-1, sum(action_dims))``,
        # so it needs one flat concatenated mask.  A ragged list of per-head
        # arrays raises before it ever reaches the policy.
        flat_masks = np.concatenate(per_head)

        action = None
        if model is not None:
            try:
                action, _ = model.predict(obs, action_masks=flat_masks, deterministic=False)
            except Exception:
                if not self._ai_model_failed:
                    self._ai_model_failed = True
                    logger.exception(
                        "AI model inference failed — falling back to random legal play."
                    )
                action = None
        if action is None:
            action = self._random_legal_action(per_head)

        return np.atleast_1d(action)

    @staticmethod
    def _head_masks(env: ContainerJaxEnv, state: EnvState) -> list[Any]:
        """Per-head action masks, in the head order the policy expects."""
        masks_dict = env.func_env._action_masks(state, env.func_env.params)
        return [masks_dict[k] for k in _MASK_HEADS]

    @staticmethod
    def _random_legal_action(per_head: list[Any]) -> np.ndarray:
        """Uniformly sample one legal index per head (0 if a head has none)."""
        choices = []
        for mask in per_head:
            legal = np.flatnonzero(mask)
            choices.append(int(np.random.choice(legal)) if legal.size else 0)
        return np.asarray(choices, dtype=np.int64)

    def play_ai_turn_if_needed(self, game_id: int) -> None:
        """Public entry point: auto-play AI turns if it's an AI's turn.

        Called after ``get_state`` (rejoin) to ensure AI players
        take action immediately instead of waiting for a human.
        """
        with self._lock:
            if game_id not in self._envs:
                return
            self._play_ai_turns(game_id)

    def _play_ai_turns(self, game_id: int) -> None:
        """Continuously play AI turns until a human player's turn or game over.

        Called after ``maybe_start_game`` and after every ``process_action``.
        """
        env = self._envs.get(game_id)
        if env is None:
            return

        ai_slots = self._get_ai_slots(game_id)
        if not ai_slots:
            return

        model = self._load_ai_model()
        nc = env.func_env.params.num_colors
        np_ = env.func_env.params.num_players
        ms_size = self._get_mask_size(np_, nc)

        # Keep playing as long as it's an AI player's turn.  The cap is a
        # backstop: a model that keeps picking non-turn-ending actions must
        # not spin forever while holding the manager lock.
        for _ in range(MAX_AI_TURN_STEPS):
            state = env.state
            if int(state.game_over) > 0:
                break

            # An open auction comes first: bidding is the one move made out of
            # turn, so the seat that opened it stays ``current_player`` and the
            # loop below would walk straight past every AI that owes a bid.
            # Nothing then arrived to close the round, and the seller's client
            # waited on it for good.
            if int(state.auction_active) > 0:
                if self._play_ai_auction_bids(game_id, model, ms_size):
                    self._broadcast_state(game_id)
                state = env.state
                if int(state.auction_active) > 0 and (
                        int(state.auction_round) == 0
                        or int(state.auction_seller) not in ai_slots):
                    # Either a human still owes a bid or a human has to accept
                    # or reject.  Stop here in both cases: a turn taken now
                    # would be the AI acting inside somebody else's auction,
                    # and the bidder head would let it bid in their name.
                    break

            cp = int(state.current_player)
            if cp not in ai_slots:
                break

            # ── 1. Choose and apply one action ──
            _obs, _reward, term, _trunc, _info = self._ai_step(env, model, ms_size)
            self._save_state(game_id, env.state)

            # ── 2. Handle shopping / produce / auction continuation ──
            self._play_ai_continuation(game_id, model, nc, np_, ms_size)

            # ── 3. Broadcast updated state ──
            new_state = env.state
            self._broadcast_state(game_id)

            # ── 4. Check for game over ──
            if bool(term) or int(new_state.game_over) > 0:
                self.db.set_game_status(game_id, "finished")
                break

    def _broadcast_state(self, game_id: int) -> None:
        """Send the current position to everyone at the table."""
        state = self._envs[game_id].state
        try:
            state_data = _state_to_json_data(state)
        except Exception:
            state_data = {}
        self._broadcast(game_id, "state_update", {
            "state": serialize_state(state).hex(),
            "state_data": state_data,
            "current_player": int(state.current_player),
            "actions_taken": int(state.actions_taken),
            "auction_active": int(state.auction_active),
            "produce_active": int(state.produce_active),
            "shopping_active": int(state.shopping_active),
            "game_over": int(state.game_over),
        })

    def _play_ai_auction_bids(self, game_id: int, model: Any, ms_size: int) -> bool:
        """Enter the bids the AI seats owe in an open auction.  True if any.

        Only the AI's own seats are bid for.  The bidder is named by a raw
        player index on the opponent head and the env does not check it
        against whoever sent the action, so an ordinary AI turn taken during
        an auction can enter a bid for a human and spend their money -- which
        is what the old ``_ai_step`` loop here did whenever an AI opened one.
        """
        env = self._envs.get(game_id)
        if env is None:
            return False
        ai_slots = self._get_ai_slots(game_id)
        if not ai_slots:
            return False

        bid_placed = False
        for _ in range(MAX_AI_CONTINUATION_STEPS):
            state = env.state
            if int(state.auction_active) == 0 or int(state.auction_round) != 0:
                break
            seller = int(state.auction_seller)
            owing = [p for p in ai_slots
                     if p != seller and int(state.auction_bids[p]) < 0]
            if not owing:
                break
            for player in owing:
                env.step(np.array(
                    [ACTION_MOVE_AUCTION + 1, player, 0, 0,
                     self._ai_bid(env, model, ms_size, player)],
                    dtype=np.int64))
                bid_placed = True
            self._save_state(game_id, env.state)
        return bid_placed

    def _ai_bid(self, env: ContainerJaxEnv, model: Any, ms_size: int,
                player: int) -> int:
        """What *player* bids, asked of the model from that seat's chair.

        Both the observation and the masks are keyed to ``current_player``, so
        the seat is swapped in for the question only -- the real state is left
        exactly as it is and stepped separately with an explicit bidder.
        """
        state = env.state
        view = state._replace(
            current_player=jnp.asarray(player, dtype=state.current_player.dtype))
        action = self._ai_choose(env, view, model, ms_size)
        bid = int(np.atleast_1d(action)[HEAD_PURCHASE])
        # The env clips to the bidder's cash; do it here too so what is
        # recorded in the log is what actually happens.
        return max(0, min(bid, int(state.cash[player])))

    def _play_ai_continuation(self, game_id: int, model: Any,
                               nc: int, np_: int, ms_size: int) -> None:
        """Handle shopping / produce / auction for an AI player."""
        env = self._envs[game_id]
        state = env.state

        steps = 0
        while (bool(state.shopping_active) or bool(state.produce_active)) \
                and steps < MAX_AI_CONTINUATION_STEPS:
            self._ai_step(env, model, ms_size)
            self._save_state(game_id, env.state)
            state = env.state
            steps += 1

        # An auction the AI just opened: collect our own bids, then close it
        # if the seller is one of ours.  Any human bidder is left to answer in
        # their own time -- the auction stays open until they do.
        if bool(env.state.auction_active):
            self._play_ai_auction_bids(game_id, model, ms_size)
            state = env.state
            steps = 0
            while bool(state.auction_active) and int(state.auction_round) == 1 \
                    and int(state.auction_seller) in self._get_ai_slots(game_id) \
                    and steps < MAX_AI_CONTINUATION_STEPS:
                self._ai_step(env, model, ms_size)
                self._save_state(game_id, env.state)
                state = env.state
                steps += 1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _save_state(self, game_id: int, state: EnvState) -> None:
        blob = serialize_state(state)
        self.db.save_state(game_id, blob, int(state.step_count))


def _state_to_json_data(state: EnvState) -> dict:
    """Convert EnvState fields to JSON-serializable Python types."""
    import numpy as np
    return {
        "cash": np.asarray(state.cash).tolist(),
        "loans": np.asarray(state.loans).tolist(),
        "factory_colors": np.asarray(state.factory_colors).tolist(),
        "warehouse_count": np.asarray(state.warehouse_count).tolist(),
        "factory_store": np.asarray(state.factory_store).tolist(),
        "harbour_store": np.asarray(state.harbour_store).tolist(),
        "island_store": np.asarray(state.island_store).tolist(),
        "ship_contents": np.asarray(state.ship_contents).tolist(),
        "ship_location": np.asarray(state.ship_location).tolist(),
        "container_supply": np.asarray(state.container_supply).tolist(),
        "turn_phase": int(state.turn_phase),
        "current_player": int(state.current_player),
        "game_over": int(state.game_over),
        "auction_active": int(state.auction_active),
        "auction_seller": int(state.auction_seller),
        "auction_cargo": np.asarray(state.auction_cargo).tolist(),
        "auction_bids": np.asarray(state.auction_bids).tolist(),
        "auction_round": int(state.auction_round),
        "actions_taken": int(state.actions_taken),
        "produced_this_turn": np.asarray(state.produced_this_turn).tolist(),
        "shopping_active": int(state.shopping_active),
        "shopping_action_type": int(state.shopping_action_type),
        "shopping_target": np.asarray(state.shopping_target).tolist(),
        "produce_active": int(state.produce_active),
        "produce_pending": np.asarray(state.produce_pending).tolist(),
        "produce_was_produced": np.asarray(state.produce_was_produced).tolist(),
    }


def _describe_action(action, encoder: ActionEncoder, num_colors: int, state=None,
                     player_names: dict[int, str] | None = None,
                     actor: int | None = None) -> str:
    """Return a human-readable description of *action* (flat int or multi-head array)."""
    try:
        if hasattr(action, "ndim") and action.ndim == 1 and action.shape[0] >= 5:
            # Every head reserves index 0 for no-op, so each one decodes as
            # ``head - 1``.  The purchase head is the exception: 0 = no-op,
            # 1-30 = values, PURCHASE_STOP = stop.
            atype = int(action[0]) - 1
            purchase = int(action[4])
            nc = num_colors
            colors = ['Red','Green','Blue','Yellow','Purple']
            opp_idx = int(action[1]) - 1  # HEAD_OPPONENT
            color = max(int(action[2]) - 1, 0)
            if atype == ACTION_PASS:
                return "Pass"
            elif atype == ACTION_BUY_FACTORY:
                return f"Buy {colors[min(color,nc-1)]} factory"
            elif atype == ACTION_BUY_WAREHOUSE:
                return "Buy warehouse"
            elif atype == ACTION_PRODUCE:
                slot = int(action[3]) - 1
                if slot < 0:
                    return "Produce"  # the opening action, before any factory
                if slot >= LEAVE_IDLE:
                    return f"Leave {colors[min(color,nc-1)]} idle"
                return f"Produce {colors[min(color,nc-1)]} at ${slot+1}"
            elif atype == ACTION_BUY_FROM_FACTORY_STORE:
                if purchase >= PURCHASE_STOP:
                    return "Stop buying"
                opp_name = _opponent_name(opp_idx, state, player_names, actor) if state else f"Player {opp_idx+1}"
                if int(action[2]) == 0:
                    return f"Buy from {opp_name}'s factory"
                return (f"Buy {colors[min(color,nc-1)]} from {opp_name}'s factory "
                        f"(harbour ${min(max(purchase + 1, 2), 6)})")
            elif atype == ACTION_MOVE_LOAD:
                if purchase >= PURCHASE_STOP:
                    return "Stop loading"
                opp_name = _opponent_name(opp_idx, state, player_names, actor) if state else f"Player {opp_idx+1}"
                if int(action[2]) == 0:
                    return f"Load from {opp_name}'s harbour"
                return f"Load {colors[min(color,nc-1)]} from {opp_name}'s harbour"
            elif atype == ACTION_MOVE_SEA:
                return "Move to sea"
            elif atype == ACTION_MOVE_AUCTION:
                return f"Auction (bid/decide: {purchase})"
            elif atype == ACTION_TAKE_LOAN:
                return "Take loan"
            elif atype == ACTION_REPAY_LOAN:
                return "Repay loan"
            return _ACTION_NAMES.get(atype, f"Action {atype}")
    except Exception:
        pass
    # Flat int fallback
    try:
        atype, params = encoder.decode(int(action))
        return _ACTION_NAMES.get(atype, f"Action {atype}")
    except Exception:
        return "Action"


def _opponent_name(opp_idx: int, state, player_names: dict[int, str] | None = None,
                   actor: int | None = None) -> str:
    """Return the target player's name (or 'Player {n}' as fallback).

    *actor* is the player who took the action; it defaults to whoever is
    up now, which is only the same player mid-turn.
    """
    player = int(state.current_player) if actor is None else int(actor)
    np_ = state.cash.shape[0]
    target = (player + opp_idx + 1) % np_
    if player_names and target in player_names:
        return player_names[target]
    return f"Player {target + 1}"
