"""Game lifecycle manager — create, join, start, and orchestrate turns."""

from __future__ import annotations

import threading
from typing import Any, Callable

import jax
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
    ActionEncoder,
    ContainerJaxEnv,
    EnvState,
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


class GameManager:
    def __init__(self, db: Database, broadcast: Callable[[int, str, Any], None]):
        self.db = db
        self._broadcast = broadcast
        self._envs: dict[int, ContainerJaxEnv] = {}
        self._encoders: dict[int, ActionEncoder] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # create / join
    # ------------------------------------------------------------------

    def create_game(
        self, player_name: str, password: str | None,
        num_players: int, num_colors: int = 5, seed: int | None = None,
    ) -> dict:
        """Create a new game and join as the first player (slot 0)."""
        player_id = self.db.upsert_player(player_name, password)
        game_id, code = self.db.create_game(num_players, num_colors, seed)
        self.db.assign_player_slot(game_id, player_id, 0)
        return {"game_id": game_id, "code": code, "player_index": 0}

    def join_game(
        self, player_name: str, password: str | None, code: str,
    ) -> dict:
        """Join an existing lobby game."""
        game = self.db.get_game_by_code(code)
        if not game:
            raise ValueError(f"Game '{code}' not found.")
        if game["status"] != "lobby":
            raise ValueError(f"Game '{code}' is not accepting players (status: {game['status']}).")
        player_id = self.db.upsert_player(player_name, password)
        player_index = self.db.assign_player_slot(game_id=game["id"], player_id=player_id)
        return {
            "game_id": game["id"],
            "code": code,
            "player_index": player_index,
            "num_players": game["num_players"],
            "num_colors": game["num_colors"],
        }

    def list_joinable(self) -> list[dict]:
        return self.db.list_joinable_games()

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
            if actual != player_index:
                return {
                    "turn_ended": False,
                    "desc": f"Not your turn (current player is {actual}).",
                    "reward": 0.0,
                    "game_over": False,
                    "error": True,
                }

            encoder = self.get_encoder(game_id)
            obs, reward, term, trunc, info = env.step(action)
            self._save_state(game_id, env.state)

            nc = encoder.num_colors
            desc = _describe_action(action, encoder, nc)

            new_state = env.state
            turn_ended = int(new_state.current_player) != actual
            game_over = bool(term) or int(new_state.game_over) > 0

            if game_over:
                self.db.set_game_status(game_id, "finished")

            # Broadcast to all connected players for this game
            state_blob = serialize_state(new_state)
            self._broadcast(game_id, "state_update", {
                "state": state_blob.hex(),
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

            return {
                "turn_ended": turn_ended,
                "desc": desc,
                "reward": float(reward),
                "game_over": game_over,
            }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _save_state(self, game_id: int, state: EnvState) -> None:
        blob = serialize_state(state)
        self.db.save_state(game_id, blob, int(state.step_count))


def _describe_action(action, encoder: ActionEncoder, num_colors: int) -> str:
    """Return a human-readable description of *action* (flat int or multi-head array)."""
    import jax.numpy as jnp
    try:
        if hasattr(action, "ndim") and action.ndim == 1 and action.shape[0] >= 5:
            atype = int(action[0])
            purchase = int(action[4])
            nc = num_colors
            if atype == ACTION_PASS:
                return "Pass"
            elif atype == ACTION_BUY_FACTORY:
                return f"Buy {['Red','Green','Blue','Yellow','Purple'][min(int(action[2]),nc-1)]} factory"
            elif atype == ACTION_BUY_WAREHOUSE:
                return "Buy warehouse"
            elif atype == ACTION_PRODUCE:
                color = int(action[2])
                slot = int(action[3])
                if slot >= LEAVE_IDLE:
                    return f"Leave {['Red','Green','Blue','Yellow','Purple'][min(color,nc-1)]} idle"
                return f"Produce {['Red','Green','Blue','Yellow','Purple'][min(color,nc-1)]} at ${slot+1}"
            elif atype == ACTION_BUY_FROM_FACTORY_STORE:
                if purchase >= nc * PRICE_SLOTS:
                    return "Stop buying"
                color = purchase // PRICE_SLOTS
                slot = purchase % PRICE_SLOTS
                return f"Buy {['Red','Green','Blue','Yellow','Purple'][min(color,nc-1)]} from factory at ${slot+1}"
            elif atype == ACTION_MOVE_LOAD:
                if purchase >= nc * PRICE_SLOTS:
                    return "Stop loading"
                color = purchase // PRICE_SLOTS
                slot = purchase % PRICE_SLOTS
                return f"Load {['Red','Green','Blue','Yellow','Purple'][min(color,nc-1)]} from harbour at ${slot+1}"
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
