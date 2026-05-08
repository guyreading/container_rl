"""TCP client for connecting to the game server."""

from __future__ import annotations

import socket
from typing import Any

from container_rl.server.protocol import pack_message, recv_message


class GameClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def disconnect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send(self, msg_type: str, payload: Any = None) -> None:
        if self.sock is None:
            raise ConnectionError("Not connected.")
        data = pack_message(msg_type, payload)
        self.sock.sendall(data)

    def recv(self) -> dict | None:
        if self.sock is None:
            return None
        return recv_message(self.sock)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
