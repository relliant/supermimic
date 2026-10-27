"""CPU-only reference protocol and acknowledged sender tests, without Redis."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from deployment.streaming import protocol as reference_stream, motion as send_motion
from deployment.streaming.protocol import ReferencePacket, decode


@pytest.fixture
def reference():
    rows = np.zeros((5, 71), dtype=np.float32)
    rows[:, 0] = np.arange(5)
    rows[:, 3] = 1
    rows[:, 36:65] = 2
    rows[:, 65:71] = 3
    return rows


@pytest.fixture
def packet(reference):
    return ReferencePacket("controller-1", "session-1", 0, 100.0, "start", "walk.csv", "recorded", reference[:3])


def test_round_trip_contract_and_stop(packet):
    encoded = packet.encode()
    message = json.loads(encoded)
    assert message["schema"] == "supermimic.reference"
    assert message["version"] == 1
    assert message["control_dt"] == 0.02
    restored = decode(encoded, 2, now=100.1, max_age=0.5)
    assert restored.reference.dtype == np.float32
    np.testing.assert_array_equal(restored.reference, packet.reference)
    assert restored.session_id == packet.session_id
    stop = replace(packet, state="stop", sequence=4, reference=None)
    assert decode(stop.encode(), 2, now=100, max_age=1).reference is None
    assert reference_stream.packet_key("example") == "example:reference"
    assert reference_stream.status_key("example") == "example:status"


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("version", True), ("version", 2),
    ("control_dt", 0.01), ("sequence", -1), ("sequence", True), ("sequence", 1),
    ("sent_at", float("nan")), ("sent_at", float("inf")), ("sent_at", "100"),
    ("state", "reset"), ("name", "x" * 257), ("horizon_mode", "raw-pico"),
    ("controller_id", "../unsafe"), ("session_id", "x" * 129), ("session_id", ""),
    ("reference", None), ("reference", [[0] * 71]), ("reference", [[0] * 70] * 3),
])
def test_corrupted_fields_rejected(packet, field, value):
    message = json.loads(packet.encode())
    message[field] = value
    with pytest.raises(ValueError):
        decode(json.dumps(message).encode(), 2, now=100, max_age=1)


@pytest.mark.parametrize("payload", [b"", b"null", b"[]", b"{", b"\xff", b"x" * (256 * 1024 + 1)])
def test_invalid_payloads_rejected(payload):
    with pytest.raises(ValueError):
        decode(payload, 2, now=100, max_age=1)


def test_duplicate_unknown_and_missing_fields_rejected(packet):
    encoded = packet.encode()
    with pytest.raises(ValueError):
        decode(encoded[:-1] + b',"version":1}', 2, now=100, max_age=1)
    message = json.loads(encoded)
    message["path"] = "/never/load/a/receiver/file"
    with pytest.raises(ValueError):
        decode(json.dumps(message).encode(), 2, now=100, max_age=1)
    del message["path"]
    del message["reference"]
    with pytest.raises(ValueError):
        decode(json.dumps(message).encode(), 2, now=100, max_age=1)


@pytest.mark.parametrize("quaternion", [[0, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, float("nan")]])
def test_non_normalized_quaternions_rejected(packet, quaternion):
    message = json.loads(packet.encode())
    message["reference"][0][3:7] = quaternion
    with pytest.raises(ValueError, match="finite|normalized"):
        decode(json.dumps(message).encode(), 2, now=100, max_age=1)


def test_no_dimension_padding_and_no_nonfinite_values(packet):
    with pytest.raises(ValueError, match="goal_frames"):
        decode(packet.encode(), 3, now=100, max_age=1)
    reference = packet.reference.copy()
    reference[0, 42] = np.inf
    with pytest.raises(ValueError, match="finite"):
        replace(packet, reference=reference).encode()
    reference[0, 42] = 0
    reference = np.tile(reference, (1000, 1))
    with pytest.raises(ValueError, match="256 KiB"):
        replace(packet, reference=reference).encode()


@pytest.mark.parametrize("now,max_age", [(101.01, 1), (99.49, 1), (100, 0), (float("nan"), 1), (100, float("inf"))])
def test_age_and_future_skew_rejected(packet, now, max_age):
    with pytest.raises(ValueError):
        decode(packet.encode(), 2, now=now, max_age=max_age)


def test_timestamp_boundary_and_quaternion_sign_preserved(packet):
    packet.reference[:, 3] = -1
    restored = decode(packet.encode(), 2, now=99.5, max_age=1)
    np.testing.assert_array_equal(restored.reference[:, 3], -1)
    decode(packet.encode(), 2, now=101, max_age=1)


def test_windows_keep_endpoint_velocity_only_for_recorded_rows(reference):
    original = reference.copy()
    window = send_motion.build_window(reference, 3, 3)
    np.testing.assert_array_equal(window[:2], reference[3:])
    np.testing.assert_array_equal(window[2:, :36], np.repeat(reference[-1:, :36], 2, axis=0))
    np.testing.assert_array_equal(window[2:, 36:71], np.zeros((2, 35)))
    legacy = send_motion.build_window(reference, 3, 3, hold_final_velocity=True)
    np.testing.assert_array_equal(legacy[2:], np.repeat(reference[-1:], 2, axis=0))
    np.testing.assert_array_equal(reference, original)


def test_redis_host_rejected_before_import():
    with pytest.raises(ValueError, match="local Redis"):
        reference_stream.connect_redis("remote.example")


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0

    def time(self):
        return 100 + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds


class FakeRedis:
    """A lease-aware transport with a separately acknowledged controller."""

    def __init__(self, clock, *, paused_until=0.0, acknowledge=True):
        self.clock = clock
        self.paused_until = paused_until
        self.acknowledge = acknowledge
        self.values = {}
        self.expires = {}
        self.publications = []
        self.steps = []
        self.renewals = 0
        self.pending = None
        self.controller = {
            "controller_id": "controller-1", "goal_frames": 2, "control_dt": 0.02,
            "state": "waiting", "session_id": None, "sequence": -1,
            "updated_at": clock.time(), "paused": False,
        }

    def get(self, key):
        if key.endswith(":status"):
            self.controller["updated_at"] = self.clock.time()
            self.controller["paused"] = self.clock.elapsed < self.paused_until
            if self.pending and not self.controller["paused"] and self.acknowledge:
                packet = self.pending
                if self.controller["session_id"] != packet.session_id or packet.sequence > self.controller["sequence"]:
                    self.controller.update(session_id=packet.session_id, sequence=packet.sequence,
                                           state="stopped" if packet.state == "stop" else "playing")
                    if packet.state != "stop":
                        self.steps.append(packet.sequence)
            return json.dumps(self.controller).encode()
        if self.clock.elapsed >= self.expires.get(key, float("inf")):
            self.values.pop(key, None)
        return self.values.get(key)

    def set(self, key, value, *, nx=False, ex=None):
        if nx and self.get(key) is not None:
            return False
        self.values[key] = value
        if ex is not None:
            self.expires[key] = self.clock.elapsed + ex
        return True

    def eval(self, script, number_keys, *arguments):
        keys = arguments[:number_keys]
        values = arguments[number_keys:]
        if self.get(keys[0]) != values[0]:
            return 0
        if script == send_motion._RENEW_LOCK:
            self.expires[keys[0]] = self.clock.elapsed + 3
            self.renewals += 1
        elif script == send_motion._RELEASE_LOCK:
            self.values.pop(keys[0], None)
        elif script == send_motion._PUBLISH:
            if self.controller["controller_id"] != values[1]:
                return 0
            self.pending = decode(values[2], 2, now=self.clock.time(), max_age=0.25)
            self.publications.append(self.pending)
            self.values[keys[2]] = values[2]
        else:
            raise AssertionError("Unexpected Lua script")
        return 1


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(send_motion, "time", SimpleNamespace(
        time=clock.time, monotonic=clock.monotonic, sleep=clock.sleep,
    ))
    return clock


def test_sender_exact_steps_seconds_cap_and_repeat_invocations(reference, clock):
    client = FakeRedis(clock)
    assert send_motion.send_motion(client, reference, name="walk", seconds=0.065) == 3
    assert client.steps == [0, 1, 2]
    assert [(packet.state, packet.sequence) for packet in client.publications] == [
        ("start", 0), ("frame", 1), ("frame", 2), ("stop", 3),
    ]
    assert client.get("supermimic:g1:publisher") is None
    first_session = client.publications[0].session_id
    send_motion.send_motion(client, reference, name="other", seconds=0.02)
    assert client.publications[-2].session_id != first_session
    assert client.steps == [0, 1, 2, 0]


def test_pause_refreshes_pending_packet_without_timeout_or_double_step(reference, clock):
    client = FakeRedis(clock, paused_until=1.2)
    send_motion.send_motion(client, reference, name="walk", seconds=0.02, timeout=0.05)
    starts = [packet for packet in client.publications if packet.state == "start"]
    assert len(starts) >= 10
    assert {packet.sequence for packet in starts} == {0}
    assert starts[-1].sent_at - starts[0].sent_at >= 1
    assert client.renewals >= 1
    assert client.steps == [0]


def test_ack_timeout_sends_stop_and_releases_own_lock(reference, clock):
    client = FakeRedis(clock, acknowledge=False)
    with pytest.raises(TimeoutError, match="acknowledgement"):
        send_motion.send_motion(client, reference, name="walk", timeout=0.04)
    assert client.publications[-1].state == "stop"
    assert client.get("supermimic:g1:publisher") is None
    assert client.steps == []


def test_second_publisher_and_cleanup_cannot_clobber_replacement(packet, clock):
    client = FakeRedis(clock)
    publisher = send_motion.MotionPublisher(client, "supermimic:g1", 1)
    publisher.acquire()
    publisher.wait_ready()
    publisher.send_and_wait(packet)
    other = send_motion.MotionPublisher(client, "supermimic:g1", 1)
    with pytest.raises(RuntimeError, match="Another"):
        other.acquire()
    publication_count = len(client.publications)
    client.values[publisher.lock_key] = "replacement-publisher"
    publisher.close(interrupted=True)
    assert client.get(publisher.lock_key) == "replacement-publisher"
    assert len(client.publications) == publication_count


def test_controller_replacement_prevents_cleanup_stop(packet, clock):
    client = FakeRedis(clock)
    publisher = send_motion.MotionPublisher(client, "supermimic:g1", 1)
    publisher.acquire()
    publisher.wait_ready()
    publisher.send_and_wait(packet)
    client.controller["controller_id"] = "replacement-controller"
    publication_count = len(client.publications)
    publisher.close(interrupted=True)
    assert len(client.publications) == publication_count


def test_paused_stopped_session_releases_sender_instead_of_waiting_forever(packet, clock):
    client = FakeRedis(clock, paused_until=100, acknowledge=False)
    publisher = send_motion.MotionPublisher(client, "supermimic:g1", 1)
    publisher.acquire()
    publisher.wait_ready()
    client.controller.update(state="stopped", session_id=packet.session_id, sequence=0)
    try:
        with pytest.raises(RuntimeError, match="stopped the motion session"):
            publisher.send_and_wait(replace(packet, state="frame", sequence=1))
    finally:
        publisher.close(interrupted=True)
    assert clock.elapsed < 1
    assert client.get("supermimic:g1:publisher") is None


def test_old_status_is_not_ready(clock):
    status = FakeRedis(clock).controller
    status["updated_at"] = clock.time() - 2.01
    client = SimpleNamespace(get=lambda key: json.dumps(status).encode())
    assert send_motion._read_status(client, "test") is None
    status["updated_at"] = clock.time() + 0.51
    assert send_motion._read_status(client, "test") is None
    status["updated_at"] = clock.time()
    assert send_motion._read_status(client, "test")["state"] == "waiting"


@pytest.mark.parametrize("state", ["error", "stale"])
def test_matching_session_rejection_is_not_acknowledged(packet, clock, state):
    client = FakeRedis(clock, acknowledge=False)
    publisher = send_motion.MotionPublisher(client, "supermimic:g1", 1)
    publisher.acquire()
    publisher.wait_ready()
    client.controller.update(state=state, session_id=packet.session_id, sequence=packet.sequence)
    with pytest.raises(RuntimeError, match="rejected motion"):
        publisher.send_and_wait(packet)
    publisher.close(interrupted=True)
