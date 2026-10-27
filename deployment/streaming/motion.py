"""Send one CSV motion to an already-running local reference-stream controller.

Run ``python -m deployment.send_motion --motion clip.csv``. Only one K+1
window is in flight: every sequence is acknowledged after one physical control
step. A subsequent invocation can select another CSV without reloading ONNX.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from deployment.common.contract import load_reference
from deployment.streaming.protocol import (
    CONTROL_DT, DEFAULT_CHANNEL, MAX_PAYLOAD_BYTES, ReferencePacket,
    connect_redis, packet_key, status_key,
)

_RENEW_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], 3)
end
return 0
"""
_RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""
# Check controller identity and lease in the same transaction as publication.
_PUBLISH = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then return 0 end
local raw = redis.call('get', KEYS[2])
if not raw then return 0 end
local valid, status = pcall(cjson.decode, raw)
if not valid or type(status) ~= 'table' then return 0 end
if status.controller_id ~= ARGV[2] then return 0 end
redis.call('set', KEYS[3], ARGV[3], 'EX', 2)
return 1
"""


def build_window(reference: np.ndarray, frame: int, goal_frames: int, *, hold_final_velocity: bool = False) -> np.ndarray:
    """Clamp pose to the endpoint, zeroing only synthetic tail velocities."""
    if reference.ndim != 2 or reference.shape[1] != 71 or len(reference) < 1:
        raise ValueError("reference must have shape (N, 71), N >= 1")
    if type(frame) is not int or not 0 <= frame < len(reference):
        raise ValueError("frame must be inside the reference clip")
    if type(goal_frames) is not int or goal_frames < 1:
        raise ValueError("goal_frames must be positive")
    requested = frame + np.arange(goal_frames + 1)
    window = np.asarray(reference[np.minimum(requested, len(reference) - 1)], dtype=np.float32).copy()
    if not hold_final_velocity:
        window[requested >= len(reference), 36:71] = 0
    return window


def _finite_positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _read_status(client, channel: str) -> dict | None:
    payload = client.get(status_key(channel))
    if payload is None:
        return None
    if not isinstance(payload, (bytes, str)) or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Invalid controller status payload")
    try:
        status = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid controller status JSON") from error
    fields = {"controller_id", "goal_frames", "control_dt", "state", "session_id", "sequence", "updated_at", "paused"}
    if not isinstance(status, dict) or not fields.issubset(status):
        raise ValueError("Controller status is missing required fields")
    if not isinstance(status["controller_id"], str) or not 1 <= len(status["controller_id"]) <= 128:
        raise ValueError("Invalid controller identity")
    if type(status["goal_frames"]) is not int or not 1 <= status["goal_frames"] <= MAX_PAYLOAD_BYTES // 71:
        raise ValueError("Invalid controller goal_frames")
    if type(status["control_dt"]) not in (int, float) or status["control_dt"] != CONTROL_DT:
        raise ValueError("Controller control_dt must be 0.02 seconds")
    if status["state"] not in ("waiting", "playing", "stopped", "stale", "error"):
        raise ValueError("Invalid controller state")
    if status["session_id"] is not None and not isinstance(status["session_id"], str):
        raise ValueError("Invalid controller session_id")
    if type(status["sequence"]) is not int or status["sequence"] < -1:
        raise ValueError("Invalid controller sequence")
    if type(status["paused"]) is not bool:
        raise ValueError("Controller paused must be boolean")
    updated_at = status["updated_at"]
    if type(updated_at) not in (int, float):
        raise ValueError("Controller timestamp must be finite")
    try:
        age = time.time() - float(updated_at)
    except OverflowError as error:
        raise ValueError("Controller timestamp must be finite") from error
    if not math.isfinite(age):
        raise ValueError("Controller timestamp must be finite")
    return status if -0.5 <= age <= 2.0 else None


class MotionPublisher:
    """An owned short Redis lease and bounded, pause-aware ACK transaction."""

    def __init__(self, client, channel: str, timeout: float):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.client = client
        self.channel = channel
        self.timeout = timeout
        self.owner = uuid.uuid4().hex
        self.lock_key = channel + ":publisher"
        self.controller_id: str | None = None
        self.pending: ReferencePacket | None = None
        self.last_renewal = 0.0
        self.last_progress = time.monotonic()

    def acquire(self) -> None:
        if not self.client.set(self.lock_key, self.owner, nx=True, ex=3):
            raise RuntimeError("Another motion publisher owns this channel")
        self.last_renewal = time.monotonic()

    def renew(self) -> None:
        if time.monotonic() - self.last_renewal >= 0.75:
            if not self.client.eval(_RENEW_LOCK, 1, self.lock_key, self.owner):
                raise RuntimeError("Motion publisher lease was lost")
            self.last_renewal = time.monotonic()

    def wait_ready(self) -> dict:
        deadline = time.monotonic() + self.timeout
        while True:
            self.renew()
            status = _read_status(self.client, self.channel)
            if status is not None:
                self.controller_id = status["controller_id"]
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a recent controller status")
            time.sleep(0.01)

    def publish(self, packet: ReferencePacket) -> None:
        payload = replace(packet, sent_at=time.time()).encode()
        if not self.client.eval(
            _PUBLISH, 3, self.lock_key, status_key(self.channel), packet_key(self.channel),
            self.owner, self.controller_id, payload,
        ):
            raise RuntimeError("Controller changed or motion publisher lease was lost")

    def send_and_wait(self, packet: ReferencePacket) -> None:
        self.renew()
        self.pending = packet
        self.publish(packet)
        last_publish = time.monotonic()
        previous_poll = last_publish
        unpaused_seconds = 0.0
        previously_paused = False
        while True:
            self.renew()
            status = _read_status(self.client, self.channel)
            current_poll = time.monotonic()
            if not previously_paused:
                unpaused_seconds += current_poll - previous_poll
            previous_poll = current_poll
            previously_paused = False
            if status is not None:
                if status["controller_id"] != self.controller_id:
                    raise RuntimeError("Controller restarted during motion playback")
                previously_paused = status["paused"]
                if status["session_id"] == packet.session_id:
                    if status["state"] in ("stale", "error"):
                        raise RuntimeError(f"Controller rejected motion: {status['state']}")
                    acknowledged = status["sequence"] >= packet.sequence
                    if acknowledged and (packet.state != "stop" or status["state"] == "stopped"):
                        return
                    if status["state"] == "stopped":
                        raise RuntimeError("Controller stopped the motion session")
            if unpaused_seconds >= self.timeout:
                raise TimeoutError(f"Timed out waiting for sequence {packet.sequence} acknowledgement")
            if current_poll - last_publish >= 0.1:
                self.publish(packet)
                last_publish = current_poll
            if current_poll - self.last_progress >= 1.0:
                print(f"sequence={packet.sequence} sim={packet.sequence * CONTROL_DT:.2f}s"
                      f" paused={previously_paused}", flush=True)
                self.last_progress = current_poll
            time.sleep(0.01)

    def close(self, *, interrupted: bool) -> None:
        try:
            if interrupted and self.pending is not None:
                # The publication Lua checks both identities atomically. Never
                # overwrite a replacement publisher/controller's reference.
                self.publish(replace(
                    self.pending, state="stop", sequence=self.pending.sequence + 1,
                    reference=None, sent_at=time.time(),
                ))
        except Exception:
            pass  # Cleanup must not hide the original interruption/transport error.
        finally:
            try:
                self.client.eval(_RELEASE_LOCK, 1, self.lock_key, self.owner)
            except Exception:
                pass  # An unreachable Redis lease expires automatically in 3 s.


def send_motion(client, reference: np.ndarray, *, name: str, channel: str = DEFAULT_CHANNEL,
                timeout: float = 10.0, seconds: float | None = None,
                hold_final_velocity: bool = False) -> int:
    """Play at most N-1 physical steps, then stop at the final reference."""
    number_steps = len(reference) - 1
    if seconds is not None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        number_steps = min(number_steps, math.floor(seconds / CONTROL_DT + 1e-9))
    if number_steps < 1:
        raise ValueError("Playback requires at least one 20 ms control step")
    publisher = MotionPublisher(client, channel, timeout)
    completed = False
    publisher.acquire()
    try:
        status = publisher.wait_ready()
        session_id = uuid.uuid4().hex
        for frame in range(number_steps):
            packet = ReferencePacket(
                controller_id=status["controller_id"], session_id=session_id,
                sequence=frame, sent_at=time.time(), state="start" if frame == 0 else "frame",
                name=name, horizon_mode="recorded",
                reference=build_window(reference, frame, status["goal_frames"],
                                       hold_final_velocity=hold_final_velocity),
            )
            publisher.send_and_wait(packet)
            if time.monotonic() - publisher.last_progress >= 1.0:
                print(f"steps={frame + 1}/{number_steps} sim={(frame + 1) * CONTROL_DT:.2f}s", flush=True)
                publisher.last_progress = time.monotonic()
        publisher.send_and_wait(replace(packet, sequence=number_steps, state="stop", reference=None))
        completed = True
        print(f"Finished {name}: {number_steps} steps, {number_steps * CONTROL_DT:.2f}s simulated", flush=True)
        return number_steps
    finally:
        publisher.close(interrupted=not completed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--src-fps", type=_finite_positive, default=30.0)
    parser.add_argument("--hold-final-velocity", action="store_true")
    parser.add_argument("--timeout", type=_finite_positive, default=10.0)
    parser.add_argument("--seconds", type=_finite_positive)
    arguments = parser.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    client = None
    try:
        reference = load_reference(arguments.motion, src_fps=arguments.src_fps)
        client = connect_redis(arguments.redis_host, arguments.redis_port, arguments.redis_db)
        send_motion(client, reference, name=arguments.motion.name, channel=arguments.channel,
                    timeout=arguments.timeout, seconds=arguments.seconds,
                    hold_final_velocity=arguments.hold_final_velocity)
    except KeyboardInterrupt:
        print("Motion interrupted; requested controller stop", flush=True)
        return 130
    except Exception as error:
        print(f"Motion sender stopped: {error}", flush=True)
        return 1
    finally:
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
