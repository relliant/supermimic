"""Version-2 buffered live references; version-1 CSV ACK packets stay unchanged.

All times are host Unix seconds after the adapter maps the SDK body clock.
source_time is acquisition progress, reference_time is the delayed first row,
and sent_at is publication time. Refreshing one does not refresh the others.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
import uuid

import numpy as np

from deployment.common.contract import CONTROL_DT
from deployment.streaming.protocol import (
    MAX_PAYLOAD_BYTES, ReferencePacket, _number, _unique_object, _validate_packet,
    packet_key, status_key,
)
from deployment.streaming.motion import _PUBLISH, _RELEASE_LOCK, _RENEW_LOCK


@dataclass
class LiveReferencePacket:
    controller_id: str
    session_id: str
    sequence: int
    sent_at: float
    state: str
    name: str
    source_sequence: int
    source_time: float
    reference_time: float
    reference: np.ndarray | None

    def validate(self, goal_frames: int | None = None) -> None:
        compatible = ReferencePacket(self.controller_id, self.session_id, self.sequence,
                                     self.sent_at, self.state, self.name, "recorded", self.reference)
        self.reference = _validate_packet(compatible)
        if type(self.source_sequence) is not int or not 0 <= self.source_sequence <= 2**63 - 1:
            raise ValueError("source_sequence must be a nonnegative 64-bit body sample ID")
        source_time = _number(self.source_time, "source_time")
        reference_time = _number(self.reference_time, "reference_time")
        if source_time <= 0 or reference_time <= 0 or source_time > self.sent_at + 0.05:
            raise ValueError("Live acquisition/reference times must be positive and acquisition cannot follow publication")
        if self.reference is not None:
            if len(self.reference) > 128 or (goal_frames is not None and len(self.reference) < goal_frames + 1):
                raise ValueError("Live chunk must contain K+1..128 complete reference rows")
            last_time = reference_time + (len(self.reference) - 1) * CONTROL_DT
            if last_time > source_time + 1e-5:
                raise ValueError("Buffered live rows cannot describe an unobserved future")

    def encode(self) -> bytes:
        self.validate()
        payload = json.dumps({
            "schema": "supermimic.reference", "version": 2, "control_dt": CONTROL_DT,
            "controller_id": self.controller_id, "session_id": self.session_id,
            "sequence": self.sequence, "sent_at": self.sent_at, "state": self.state, "name": self.name,
            "source_sequence": self.source_sequence, "source_time": self.source_time,
            "reference_time": self.reference_time,
            "reference": self.reference.tolist() if self.reference is not None else None,
        }, allow_nan=False, separators=(",", ":")).encode()
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("Live reference packet exceeds 256 KiB")
        return payload


def decode_live(payload: bytes, goal_frames: int, *, now: float, max_age: float,
                check_freshness: bool = True) -> LiveReferencePacket:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_PAYLOAD_BYTES:
        raise ValueError("Live payload must be bounded bytes")
    try:
        message = json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid live reference JSON") from error
    fields = set(LiveReferencePacket.__dataclass_fields__) | {"schema", "version", "control_dt"}
    if not isinstance(message, dict) or set(message) != fields:
        raise ValueError("Live packet has missing or unknown fields")
    if (message.pop("schema") != "supermimic.reference" or type(message["version"]) is not int
            or message.pop("version") != 2 or _number(message.pop("control_dt"), "control_dt") != CONTROL_DT):
        raise ValueError("Expected version-2 live reference at 20 ms")
    packet = LiveReferencePacket(**message)
    packet.validate(goal_frames)
    if check_freshness:
        validate_freshness(packet, now=now, max_age=max_age)
    return packet


def validate_freshness(packet: LiveReferencePacket, *, now: float, max_age: float) -> None:
    now = _number(now, "now")
    max_age = _number(max_age, "max_age")
    if max_age <= 0:
        raise ValueError("max_age must be positive")
    # Stopping a stale source is always permitted for the owning session;
    # freshness of its stop publication still matters.
    timestamps = (packet.sent_at,) if packet.state == "stop" else (packet.sent_at, packet.source_time)
    for timestamp in timestamps:
        age = now - timestamp
        if age > max_age or age < -0.05:
            raise ValueError("Live packet/source sample is stale or future-dated")


class LivePublisher:
    """Owner-checked latest-only publication; only start uses an ACK handshake."""

    def __init__(self, client, channel: str, timeout: float = 10):
        self.client, self.channel, self.timeout = client, channel, timeout
        self.owner = uuid.uuid4().hex
        self.lock_key = channel + ":publisher"
        self.controller_id = None
        self.last_renewal = 0.0
        self.cancel_requested = None

    def acquire(self) -> None:
        if not self.client.set(self.lock_key, self.owner, nx=True, ex=3):
            raise RuntimeError("Another producer owns this channel")
        self.last_renewal = time.monotonic()

    def _renew(self) -> None:
        if time.monotonic() - self.last_renewal >= 0.75:
            if not self.client.eval(_RENEW_LOCK, 1, self.lock_key, self.owner):
                raise RuntimeError("Live publisher lease lost")
            self.last_renewal = time.monotonic()

    def read_status(self) -> dict | None:
        self._renew()
        payload = self.client.get(status_key(self.channel))
        if payload is None:
            return None
        try:
            status = json.loads(payload)
            if (not isinstance(status, dict) or status.get("stream_mode") != "live"
                    or status.get("control_dt") != CONTROL_DT or type(status.get("goal_frames")) is not int
                    or not 1 <= status["goal_frames"] <= 100 or not isinstance(status.get("controller_id"), str)
                    or not -0.05 <= time.time() - _number(status.get("updated_at"), "updated_at") <= 2):
                raise ValueError("No recent, compatible live controller status")
        except (ValueError, TypeError, UnicodeError) as error:
            raise RuntimeError("Controller must be started with --stream-mode live") from error
        if self.controller_id and status["controller_id"] != self.controller_id:
            raise RuntimeError("Controller restarted; stop and rearm the producer")
        return status

    def wait_ready(self) -> dict:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            status = self.read_status()
            if status is not None:
                self.controller_id = status["controller_id"]
                return status
            time.sleep(0.02)
        raise TimeoutError("No live controller heartbeat")

    def publish(self, packet: LiveReferencePacket) -> None:
        self._renew()
        # Never rewrite source_time or source_sequence on retries.
        if not self.client.eval(_PUBLISH, 3, self.lock_key, status_key(self.channel), packet_key(self.channel),
                                self.owner, self.controller_id, packet.encode()):
            raise RuntimeError("Producer/controller ownership changed")

    def start(self, packet: LiveReferencePacket) -> None:
        if self.cancel_requested is not None and self.cancel_requested():
            raise RuntimeError("Live start cancelled by operator")
        self.publish(packet)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.cancel_requested is not None and self.cancel_requested():
                raise RuntimeError("Live start cancelled by operator")
            status = self.read_status()
            if status and status.get("session_id") == packet.session_id:
                if status.get("state") in ("stale", "error", "stopped") or status.get("paused"):
                    raise RuntimeError("Live start rejected; rearm with fresh body samples")
                if status.get("sequence", -1) >= packet.sequence:
                    return
            time.sleep(0.01)
        raise TimeoutError("Live start was not acknowledged")

    def close(self, packet: LiveReferencePacket | None = None) -> None:
        try:
            if packet is not None:
                self.publish(packet)
        finally:
            self.client.eval(_RELEASE_LOCK, 1, self.lock_key, self.owner)


class LiveReferenceReceiver:
    """Rolling timeline ingestion plus an independent, deterministic control tick."""

    def __init__(self, controller_id: str, goal_frames: int, *, source_timeout: float = 0.5,
                 hold_timeout: float = 0.12, max_delay: float = 1.0):
        for setting in (source_timeout, hold_timeout, max_delay):
            if not math.isfinite(setting) or setting <= 0:
                raise ValueError("Live timeouts/delay must be finite and positive")
        self.controller_id, self.goal_frames = controller_id, goal_frames
        self.source_timeout, self.hold_timeout, self.max_delay = source_timeout, hold_timeout, max_delay
        self.session_id = None
        self.sequence = -1
        self.source_sequence = -1
        self.source_time = None
        self.reference_time = None
        self.epoch = None
        self.cursor = 0
        self.rows: dict[int, np.ndarray] = {}
        self.last_window = None
        self.last_window_time = None
        self.holding_since = None
        self.last_source_progress = None
        self.source_progress_pending = False
        self.closed_sessions: set[str] = set()
        self.state, self.reason = "waiting", "waiting for buffered live start"
        self.just_started = False
        self.name = "waiting"
        self.control_steps = 0
        self.held_steps = 0
        self.session_count = 0

    def stop(self, reason: str, state: str = "stopped") -> None:
        if self.session_id:
            self.closed_sessions.add(self.session_id)
        self.rows.clear()
        self.last_window = None
        self.state, self.reason = state, reason

    def ingest(self, payload: bytes | None, simulation, *, wall_time: float) -> None:
        if payload is None:
            return
        try:
            packet = decode_live(payload, self.goal_frames, now=wall_time, max_age=self.source_timeout,
                                 check_freshness=False)
            if packet.controller_id != self.controller_id or packet.session_id in self.closed_sessions:
                return
            new_session = packet.session_id != self.session_id
            if new_session and packet.state != "start":
                return
            if not new_session and packet.sequence <= self.sequence:
                return
            validate_freshness(packet, now=wall_time, max_age=self.source_timeout)
            if packet.state == "stop":
                if not new_session:
                    self.sequence = packet.sequence
                    self.stop("live producer stopped")
                return
            if not new_session and self.state not in ("playing", "holding", "buffering"):
                return
            if not new_session and (packet.source_sequence < self.source_sequence or packet.source_time < self.source_time):
                raise ValueError("Body source clock regressed; rearm a new session")
            if (not new_session and packet.source_sequence == self.source_sequence
                    and packet.source_time != self.source_time):
                raise ValueError("A repeated body sample cannot acquire a new source timestamp")
            if not new_session and packet.source_sequence == self.source_sequence:
                return
            if not new_session and packet.source_time <= self.source_time:
                raise ValueError("New body sample must advance acquisition time")
            if new_session:
                if self.session_id:
                    self.closed_sessions.add(self.session_id)
                self.session_id, self.epoch = packet.session_id, packet.reference_time
                self.rows.clear()
                self.cursor = 0
                self.last_window = None
                self.last_window_time = None
                self.holding_since = None
                self.last_source_progress = None
                self.just_started = True
                self.session_count += 1
            offset_float = (packet.reference_time - self.epoch) / CONTROL_DT
            offset = round(offset_float)
            if abs(offset_float - offset) > 1e-3:
                raise ValueError("Live chunk is not aligned with the session's 20 ms grid")
            if offset > self.cursor or offset < -128:
                raise ValueError("Reference gap/backlog requires a new session")
            self.rows.update({offset + index: row.copy() for index, row in enumerate(packet.reference)})
            self.rows = {index: row for index, row in self.rows.items()
                         if self.cursor - 5 <= index <= self.cursor + 127}
            self.sequence, self.source_sequence = packet.sequence, packet.source_sequence
            self.source_time, self.name = packet.source_time, packet.name
            self.source_progress_pending = True
            if new_session:
                window = self._window()
                if window is None:
                    raise ValueError("Live start must contain an entire genuine lookahead window")
                self.state = "buffering"
                simulation.accept_reference(window, packet.name, reset=True)
                self.state, self.reason = "buffering", "start acknowledged; next control tick begins playback"
        except (ValueError, RuntimeError, TypeError) as error:
            # An expired old Redis key after explicit stop is harmless. An
            # active stream never treats it as fresh acquisition progress.
            if self.state not in ("waiting", "stopped", "stale", "error"):
                self.stop(str(error), "stale")
            elif self.state == "waiting":
                self.reason = str(error)

    def _window(self) -> np.ndarray | None:
        if not all(self.cursor + index in self.rows for index in range(self.goal_frames + 1)):
            return None
        return np.stack([self.rows[self.cursor + index] for index in range(self.goal_frames + 1)])

    def tick(self, simulation, *, wall_time: float, monotonic_time: float) -> bool:
        if self.state not in ("buffering", "playing", "holding"):
            return False
        if self.source_progress_pending:
            self.last_source_progress = monotonic_time
            self.source_progress_pending = False
        if (not -0.05 <= wall_time - self.source_time <= self.source_timeout
                or monotonic_time - self.last_source_progress > self.source_timeout):
            self.stop("Pico body sample stopped advancing", "stale")
            return False
        candidate_time = self.epoch + self.cursor * CONTROL_DT
        if wall_time - candidate_time > self.max_delay:
            self.stop("Live reference delay exceeded limit; rearm without replaying backlog", "stale")
            return False
        if self.just_started:
            self.just_started = False
            return False  # Publish start ACK before attempting the first physical step.
        window = self._window()
        if window is None:
            if self.last_window is None:
                return False
            if self.holding_since is None:
                self.holding_since = monotonic_time
            if monotonic_time - self.holding_since >= self.hold_timeout:
                self.stop("Lookahead underrun exceeded bounded hold", "stale")
                return False
            window = self.last_window
            self.reference_time = self.last_window_time
            self.state, self.reason = "holding", "reference cursor held; robot feedback/control continues"
            self.held_steps += 1
        else:
            self.holding_since = None
            self.last_window = window.copy()
            self.reference_time = candidate_time
            self.last_window_time = candidate_time
            self.state, self.reason = "playing", ""
        try:
            simulation.accept_reference(window, self.name, reset=False)
            simulation.advance()
        except (ValueError, RuntimeError) as error:
            self.stop(str(error), "error")
            return False
        if self.state == "playing":
            self.cursor += 1
        self.control_steps += 1
        return True

    def status(self, *, paused: bool) -> dict:
        now = time.time()
        return {
            "schema": "supermimic.reference.status", "version": 2, "stream_mode": "live",
            "controller_id": self.controller_id, "goal_frames": self.goal_frames, "control_dt": CONTROL_DT,
            "state": self.state, "reason": self.reason, "session_id": self.session_id,
            "sequence": self.sequence, "source_sequence": self.source_sequence,
            "updated_at": now, "paused": paused, "source_age_seconds": now - self.source_time if self.source_time else None,
            "reference_delay_seconds": now - self.reference_time if self.reference_time else None,
            "buffer_frames": len(self.rows), "control_steps": self.control_steps,
            "held_steps": self.held_steps, "session_count": self.session_count,
        }
