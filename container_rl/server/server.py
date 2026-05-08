"""TCP game server — accepts client connections and coordinates games.

Usage::

    python -m container_rl.server [--host 0.0.0.0] [--port 9876]
"""

from __future__ import annotations

import json
import logging
import selectors
import socket
import sys
import threading
from typing import Any

from container_rl.server.database import Database
from container_rl.server.game_manager import GameManager
from container_rl.server.protocol import pack_message, recv_message

logger = logging.getLogger("container.server")

# ---------------------------------------------------------------------------
# client handler (one per connected socket)
# ---------------------------------------------------------------------------

class ClientHandler:
    def __init__(self, sock: socket.socket, addr: tuple, server: "GameServer"):
        self.sock = sock
        self.addr = addr
        self.server = server
        self.game_id: int | None = None
        self.player_index: int | None = None
        self._recv_buf = b""

    def handle_readable(self) -> None:
        msg = recv_message(self.sock)
        if msg is None:
            self._disconnect()
            return
        self._dispatch(msg)

    def send(self, msg_type: str, payload: Any = None) -> None:
        try:
            data = pack_message(msg_type, payload)
            self.sock.sendall(data)
        except OSError:
            self._disconnect()

    def _dispatch(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})

        if msg_type == "create_game":
            self._handle_create(payload)
        elif msg_type == "join_game":
            self._handle_join(payload)
        elif msg_type == "list_games":
            self._handle_list()
        elif msg_type == "action":
            self._handle_action(payload)
        elif msg_type == "action_multi":
            self._handle_action_multi(payload)
        elif msg_type == "get_state":
            self._handle_get_state()
        elif msg_type == "heartbeat":
            self.send("heartbeat_ack", {})
        else:
            self.send("error", {"message": f"Unknown message type: {msg_type}"})

    def _handle_create(self, p: dict) -> None:
        try:
            name = p.get("player_name", "").strip()
            password = p.get("password") or None
            num_players = int(p.get("num_players", 2))
            num_colors = int(p.get("num_colors", 5))
            seed = p.get("seed")
            if seed is not None:
                seed = int(seed)
            if not name:
                raise ValueError("Player name is required.")
            if num_players < 2 or num_players > 6:
                raise ValueError("num_players must be 2–6.")

            result = self.server.manager.create_game(name, password, num_players, num_colors, seed)
            self.game_id = result["game_id"]
            self.player_index = result["player_index"]
            self.server._register_client(self)
            self.send("game_created", {
                "game_id": result["game_id"],
                "code": result["code"],
                "player_index": result["player_index"],
            })
            self._send_lobby()
        except Exception as e:
            self.send("error", {"message": str(e)})

    def _handle_join(self, p: dict) -> None:
        try:
            name = p.get("player_name", "").strip()
            password = p.get("password") or None
            code = p.get("code", "").strip().upper()
            if not name:
                raise ValueError("Player name is required.")
            if not code:
                raise ValueError("Game code is required.")

            result = self.server.manager.join_game(name, password, code)
            self.game_id = result["game_id"]
            self.player_index = result["player_index"]
            self.server._register_client(self)
            self.send("game_joined", {
                "game_id": result["game_id"],
                "code": result["code"],
                "player_index": result["player_index"],
                "num_players": result["num_players"],
                "num_colors": result["num_colors"],
            })
            self._send_lobby()

            # Check if game can start
            started = self.server.manager.maybe_start_game(self.game_id)
            if started:
                self._broadcast_to_game("game_started", {})
        except Exception as e:
            self.send("error", {"message": str(e)})

    def _handle_list(self) -> None:
        try:
            games = self.server.manager.list_joinable()
            self.send("game_list", {"games": games})
        except Exception as e:
            self.send("error", {"message": str(e)})

    def _handle_action(self, p: dict) -> None:
        if self.game_id is None or self.player_index is None:
            self.send("error", {"message": "Not in a game."})
            return
        try:
            action_idx = int(p.get("action_idx", 0))
            result = self.server.manager.process_action(self.game_id, self.player_index, action_idx)
            if result is None:
                self.send("error", {"message": "Game not found."})
                return
            self.send("action_result", result)
        except Exception as e:
            self.send("error", {"message": str(e)})

    def _handle_action_multi(self, p: dict) -> None:
        if self.game_id is None or self.player_index is None:
            self.send("error", {"message": "Not in a game."})
            return
        try:
            import jax.numpy as jnp
            arr = jnp.array(p.get("action", [0,0,0,0,0]), dtype=jnp.int32)
            result = self.server.manager.process_action(self.game_id, self.player_index, arr)
            if result is None:
                self.send("error", {"message": "Game not found."})
                return
            self.send("action_result", result)
        except Exception as e:
            self.send("error", {"message": str(e)})

    def _handle_get_state(self) -> None:
        if self.game_id is None:
            self.send("error", {"message": "Not in a game."})
            return
        try:
            state = self.server.manager.get_state(self.game_id)
            from container_rl.server.protocol import serialize_state
            blob = serialize_state(state)
            self.send("state_update", {
                "state": blob.hex(),
                "current_player": int(state.current_player),
                "actions_taken": int(state.actions_taken),
                "auction_active": int(state.auction_active),
                "produce_active": int(state.produce_active),
                "shopping_active": int(state.shopping_active),
                "game_over": int(state.game_over),
            })
        except Exception as e:
            self.send("error", {"message": str(e)})

    def _send_lobby(self) -> None:
        if self.game_id is None:
            return
        players = self.server.manager.get_game_players(self.game_id)
        game = self.server.manager.get_game_info(self.game_id)
        self.send("lobby_update", {
            "players": players,
            "num_players_needed": game["num_players"],
            "code": game["code"],
        })

    def _broadcast_to_game(self, msg_type: str, payload: Any) -> None:
        if self.game_id is None:
            return
        self.server.broadcast(self.game_id, msg_type, payload)

    def _disconnect(self) -> None:
        logger.info("Client disconnected: %s", self.addr)
        if self.game_id is not None:
            self.server._unregister_client(self)
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# game server
# ---------------------------------------------------------------------------

class GameServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876, db_path: str = "container_server.db"):
        self.host = host
        self.port = port
        self.db = Database(db_path)
        self.manager = GameManager(self.db, self.broadcast)
        self._game_clients: dict[int, list[ClientHandler]] = {}  # game_id -> clients
        self._lock = threading.Lock()
        self._running = False

    def broadcast(self, game_id: int, msg_type: str, payload: Any) -> None:
        """Send a message to all connected clients for *game_id*."""
        with self._lock:
            clients = list(self._game_clients.get(game_id, []))
        for client in clients:
            client.send(msg_type, payload)

    def _register_client(self, client: ClientHandler) -> None:
        with self._lock:
            gid = client.game_id
            if gid is not None:
                self._game_clients.setdefault(gid, []).append(client)

    def _unregister_client(self, client: ClientHandler) -> None:
        with self._lock:
            gid = client.game_id
            if gid is not None and gid in self._game_clients:
                lst = self._game_clients[gid]
                if client in lst:
                    lst.remove(client)
                if not lst:
                    del self._game_clients[gid]

    def run(self) -> None:
        """Start the server and block until interrupted."""
        sel = selectors.DefaultSelector()
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(128)
        server_sock.setblocking(False)
        sel.register(server_sock, selectors.EVENT_READ, data=None)

        self._running = True
        logger.info("Server listening on %s:%d", self.host, self.port)
        print(f"Container game server listening on {self.host}:{self.port}")
        print("Press Ctrl+C to stop.")

        try:
            while self._running:
                events = sel.select(timeout=1.0)
                for key, mask in events:
                    if key.data is None:
                        # New connection
                        conn, addr = key.fileobj.accept()
                        conn.setblocking(False)
                        logger.info("New connection from %s", addr)
                        handler = ClientHandler(conn, addr, self)
                        sel.register(conn, selectors.EVENT_READ, data=handler)
                    else:
                        handler = key.data
                        try:
                            handler.handle_readable()
                        except (ConnectionError, OSError) as e:
                            logger.warning("Error from %s: %s", handler.addr, e)
                            sel.unregister(key.fileobj)
                            handler._disconnect()
        except KeyboardInterrupt:
            logger.info("Server shutting down.")
        finally:
            sel.unregister(server_sock)
            server_sock.close()
            self._running = False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Container game server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9876, help="TCP port")
    parser.add_argument("--db", default="container_server.db", help="SQLite database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    server = GameServer(host=args.host, port=args.port, db_path=args.db)
    server.run()


if __name__ == "__main__":
    main()
