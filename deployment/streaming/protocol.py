"""Version-1 CSV/ACK reference protocol, independent of ONNX and simulation.

Each start/frame supplies CURRENT + K future rows at 20 ms intervals. Rows are
position(3), normalized quaternion_wxyz(4), joints(29), joint velocities(29),
world linear velocity(3), world angular velocity(3). Receivers load no paths or
files from packets. Buffered Pico acquisition uses the separately versioned
protocol in live_reference.py; its rows share this same full-state layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

import numpy as np

CONTROL_DT = 0.02
REFERENCE_DIM = 71
DEFAULT_CHANNEL = "supermimic:g1"
MAX_PAYLOAD_BYTES = 256 * 1024
SCHEMA = "supermimic.reference"
VERSION = 1
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_FIELDS = {
    "schema", "version", "control_dt", "controller_id", "session_id", "sequence",
    "sent_at", "state", "name", "horizon_mode", "reference",
}


class ExpiredReferenceError(ValueError):
    """A correctly formed packet is no longer fresh enough to execute."""


def packet_key(channel: str) -> str:
    return channel + ":reference"


def status_key(channel: str) -> str:
    return channel + ":status"


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_packet(packet: ReferencePacket, goal_frames: int | None = None) -> np.ndarray | None:
    for name in ("controller_id", "session_id"):
        identifier = getattr(packet, name)
        if not isinstance(identifier, str) or _SAFE_IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError(f"{name} must be a safe 1..128 character identifier")
    if type(packet.sequence) is not int or not 0 <= packet.sequence <= 2**63 - 1:
        raise ValueError("sequence must be a nonnegative 64-bit integer")
    _number(packet.sent_at, "sent_at")
    if packet.state not in ("start", "frame", "stop"):
        raise ValueError("state must be start, frame, or stop")
    if packet.state == "start" and packet.sequence != 0:
        raise ValueError("start requires sequence zero")
    if not isinstance(packet.name, str) or len(packet.name) > 256:
        raise ValueError("name must be a string of at most 256 characters")
    if packet.horizon_mode not in ("recorded", "hold", "predicted"):
        raise ValueError("horizon_mode must be recorded, hold, or predicted")
    if packet.reference is None:
        if packet.state != "stop":
            raise ValueError("start/frame require a reference window")
        return None
    try:
        reference = np.asarray(packet.reference)
        if reference.dtype.kind not in "fiu":
            raise ValueError("reference must contain numeric values")
        if reference.ndim != 2 or reference.shape[1] != REFERENCE_DIM or reference.shape[0] < 2:
            raise ValueError("reference must have shape (K+1, 71), K >= 1")
        if goal_frames is not None and reference.shape[0] != goal_frames + 1:
            raise ValueError("reference window does not match goal_frames")
        with np.errstate(over="ignore", invalid="ignore"):
            reference = reference.astype(np.float32)
    except (TypeError, OverflowError, ValueError) as error:
        raise ValueError(f"Invalid reference window: {error}") from error
    if not np.isfinite(reference).all():
        raise ValueError("reference must contain only finite float32 values")
    quaternion_norm = np.linalg.norm(reference[:, 3:7].astype(np.float64), axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0, atol=1e-4):
        raise ValueError("reference wxyz quaternions must be normalized")
    return reference


@dataclass
class ReferencePacket:
    controller_id: str
    session_id: str
    sequence: int
    sent_at: float
    state: str
    name: str
    horizon_mode: str
    reference: np.ndarray | None

    def encode(self) -> bytes:
        """Validate and serialize a bounded, plain JSON message."""
        reference = _validate_packet(self)
        message = {
            "schema": SCHEMA, "version": VERSION, "control_dt": CONTROL_DT,
            "controller_id": self.controller_id, "session_id": self.session_id,
            "sequence": self.sequence, "sent_at": self.sent_at, "state": self.state,
            "name": self.name, "horizon_mode": self.horizon_mode,
            "reference": None if reference is None else reference.tolist(),
        }
        payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("reference packet exceeds 256 KiB")
        return payload


def encode(packet: ReferencePacket) -> bytes:
    return packet.encode()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def decode(payload: bytes, goal_frames: int, *, now: float, max_age: float) -> ReferencePacket:
    """Reject malformed, incompatible, stale, and future-dated packets."""
    if type(goal_frames) is not int or goal_frames < 1:
        raise ValueError("goal_frames must be a positive integer")
    current_time = _number(now, "now")
    allowed_age = _number(max_age, "max_age")
    if allowed_age <= 0:
        raise ValueError("max_age must be positive")
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_PAYLOAD_BYTES:
        raise ValueError("payload must be bytes of at most 256 KiB")
    try:
        message = json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid reference JSON") from error
    if not isinstance(message, dict) or set(message) != _FIELDS:
        raise ValueError("reference packet has missing or unknown fields")
    if message["schema"] != SCHEMA or type(message["version"]) is not int or message["version"] != VERSION:
        raise ValueError("Unsupported reference schema/version")
    if _number(message["control_dt"], "control_dt") != CONTROL_DT:
        raise ValueError("control_dt must be 0.02 seconds")
    packet = ReferencePacket(**{name: message[name] for name in _FIELDS - {"schema", "version", "control_dt"}})
    packet.reference = _validate_packet(packet, goal_frames)
    age = current_time - packet.sent_at
    if age > allowed_age:
        raise ExpiredReferenceError("reference packet is stale")
    if age < -0.5:
        raise ValueError("reference packet is future-dated")
    return packet


def connect_redis(host: str = "127.0.0.1", port: int = 6379, database: int = 0):
    """Create a short-timeout local client, without connecting or retrying."""
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Reference streaming only supports local Redis hosts")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Redis port must be in 1..65535")
    if type(database) is not int or database < 0:
        raise ValueError("Redis database must be a nonnegative integer")
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    return redis.Redis(
        host=host, port=port, db=database, socket_timeout=0.1,
        socket_connect_timeout=0.1, retry=Retry(NoBackoff(), 0),
        retry_on_error=[],
    )
