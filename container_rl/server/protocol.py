"""Wire protocol between server and clients.

Messages are length-prefixed JSON over TCP.  Each frame is::

    [4-byte big-endian length] [JSON bytes]

The JSON envelope is::

    {"type": "<message_type>", "payload": {...}}
"""

from __future__ import annotations

import json
import pickle
import struct
from typing import Any

import jax.numpy as jnp
import numpy as np
from container_rl.env.container import EnvState


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------

def pack_message(msg_type: str, payload: Any = None) -> bytes:
    """Encode a message as a length-prefixed JSON frame."""
    if payload is None:
        payload = {}
    body = json.dumps({"type": msg_type, "payload": payload}).encode("utf-8")
    header = struct.pack(">I", len(body))
    return header + body


def recv_message(sock) -> dict | None:
    """Read one length-prefixed JSON frame from *sock*.

    Returns the parsed dict or *None* if the connection closed cleanly.
    Raises :exc:`ConnectionError` on malformed data.
    """
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    if length > 10 * 1024 * 1024:  # 10 MB sanity cap
        raise ConnectionError(f"Frame too large: {length} bytes")
    body = _recv_exact(sock, length)
    if body is None:
        raise ConnectionError("Connection closed mid-frame")
    return json.loads(body.decode("utf-8"))


def _recv_exact(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# state serialization
# ---------------------------------------------------------------------------

_STATE_KEYS = [
    "cash", "loans", "factory_colors", "warehouse_count",
    "factory_store", "harbour_store", "island_store",
    "ship_contents", "ship_location", "container_supply",
    "turn_phase", "current_player", "game_over", "secret_card_values",
    "auction_active", "auction_seller", "auction_cargo",
    "auction_bids", "auction_round",
    "actions_taken", "produced_this_turn",
    "shopping_active", "shopping_action_type", "shopping_target",
    "shopping_harbour_price",
    "produce_active", "produce_pending", "produce_was_produced",
    "step_count",
]

# Fields that are 0-d scalars.  Only these can be safely back-filled when an
# older save predates them — zero is a valid "unset" value and the shape is
# unambiguous.  Anything else (a per-player or per-colour array) would need a
# shape we cannot infer, so a missing one is an error rather than a zero.
_SCALAR_KEYS = frozenset({
    "turn_phase", "current_player", "game_over",
    "auction_active", "auction_seller", "auction_round",
    "actions_taken", "produced_this_turn",
    "shopping_active", "shopping_action_type", "shopping_target",
    "shopping_harbour_price",
    "produce_active", "produce_was_produced",
    "step_count",
})


def serialize_state(state: EnvState) -> bytes:
    """Convert an *EnvState* to a pickle-able bytes object."""
    data = {key: np.asarray(getattr(state, key)) for key in _STATE_KEYS}
    return pickle.dumps(data)


def deserialize_state(blob: bytes) -> EnvState:
    """Reconstruct an *EnvState* from pickled bytes.

    Scalar fields added since the save was written are back-filled with zero.
    A missing *array* field is fatal: we cannot infer its shape, and filling a
    0-d zero produces a state that loads cleanly and then fails much later on
    the first index into it.

    Raises:
        ValueError: the blob predates a non-scalar field of the current
            ``EnvState`` (e.g. a save written before ``secret_value_color``
            became the per-colour ``secret_card_values``).
    """
    data = pickle.loads(blob)
    kwargs = {}
    missing = []
    for key in _STATE_KEYS:
        if key in data:
            kwargs[key] = jnp.asarray(data[key])
        elif key in _SCALAR_KEYS:
            kwargs[key] = jnp.zeros((), dtype=jnp.int32)
        else:
            missing.append(key)
    if missing:
        raise ValueError(
            "Saved game state is from an incompatible older format; missing "
            f"field(s): {', '.join(missing)}. Start a new game."
        )
    return EnvState(**kwargs)
