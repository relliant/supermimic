"""Hardware-free Pico adapter, clock, mailbox, and fail-closed session tests."""

from dataclasses import dataclass
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from deployment import teleop_pico as pico


@dataclass
class Packet:
    controller_id: str
    session_id: str
    sequence: int
    sent_at: float
    state: str
    name: str
    source_sequence: int
    source_time: float
    reference_time: float
    reference: object


class Publisher:
    def __init__(self):
        self.starts = []
        self.frames = []
        self.closed = []

    def start(self, packet):
        self.starts.append(packet)

    def publish(self, packet):
        self.frames.append(packet)

    def close(self, packet=None):
        self.closed.append(packet)


class Buffer:
    def __init__(self):
        self.clear_count = 0
        self.samples = []
        self.source_time = None
        self.source_sequence = None

    def clear(self):
        self.clear_count += 1
        self.samples.clear()

    def add(self, qpos, *, source_sequence, source_time):
        self.samples.append(qpos)
        self.source_sequence = source_sequence
        self.source_time = source_time
        return True

    def chunk(self):
        if len(self.samples) < 2:
            return None
        rows = np.zeros((4, 71))
        rows[:, 3] = 1
        return self.source_time - 0.1, rows


@pytest.fixture
def session():
    retargeter = SimpleNamespace(retarget=lambda body: np.zeros(36))
    return pico.TeleopSession(Publisher(), Buffer(), retargeter, Packet, "controller")


def sample(sequence, monotonic_time=1.0):
    return pico.BodySample(sequence, 100 + sequence * 1e-9, monotonic_time, {})


def status(session=None, **updates):
    result = {"controller_id": "controller", "stream_mode": "live", "paused": False,
              "state": "waiting" if session is None else "playing",
              "session_id": None if session is None else session.session_id}
    result.update(updates)
    return result


def step(session, sequence, *, right_a=False, left_x=False, now=1.0, generation=0, receiver=None):
    return session.step(right_a=right_a, left_x=left_x, sample=sample(sequence, now),
                        generation=generation, status=status(session) if receiver is None else receiver, now=now)


def start_session(session):
    step(session, 1_000_000_000, right_a=True)
    step(session, 1_020_000_000, right_a=False, now=1.02)
    step(session, 1_040_000_000, now=1.04)
    assert len(session.publisher.starts) == 1


def test_clock_maps_arbitrary_origin_and_does_not_refresh_duplicates():
    clock = pico.BodyClock()
    assert clock.map(9_000_000_000, unix_time=1000, monotonic_time=10) == 1000
    assert clock.map(9_020_000_000, unix_time=2000, monotonic_time=10.02) == pytest.approx(1000.02)
    assert clock.map(9_020_000_000, unix_time=3000, monotonic_time=10.4) is None


@pytest.mark.parametrize("sequence,match", [(999_999_999, "regressed"),
                                         (1_200_000_000, "gap"),
                                         (True, "integer"), (0, "integer")])
def test_clock_rejects_regressions_gaps_and_invalid_timestamps(sequence, match):
    clock = pico.BodyClock()
    clock.map(1_000_000_000, unix_time=100, monotonic_time=1)
    with pytest.raises(pico.SourceDiscontinuity, match=match):
        clock.map(sequence, unix_time=100.02, monotonic_time=1.02)


def test_tiny_positive_timestamp_updates_do_not_disarm_or_refresh_freshness():
    clock = pico.BodyClock()
    clock.map(1_000_000_000, unix_time=100, monotonic_time=1)
    for increment in (1, 100, 9_999):
        assert clock.map(1_000_000_000 + increment, unix_time=100.01, monotonic_time=1.01) is None
        assert clock.previous_sequence == 1_000_000_000
        assert clock.previous_monotonic == 1
        assert clock.mapped_time == 100
    assert clock.map(1_020_000_000, unix_time=100.02, monotonic_time=1.02) == pytest.approx(100.02)


def test_clock_rejects_elapsed_clock_divergence():
    clock = pico.BodyClock()
    clock.map(1_000_000_000, unix_time=100, monotonic_time=1)
    with pytest.raises(pico.SourceDiscontinuity, match="diverged"):
        clock.map(1_020_000_000, unix_time=101, monotonic_time=2)


def test_clock_slews_normal_oscillator_drift_without_refreshing_old_body():
    clock = pico.BodyClock()
    for index in range(5001):
        mapped = clock.map(1_000_000_000 + index * 20_002_000,
                           unix_time=1000 + index * .02, monotonic_time=10 + index * .02)
    assert abs(mapped - 1100) < .001
    assert clock.map(1_000_000_000 + 5000 * 20_002_000,
                     unix_time=2000, monotonic_time=1000) is None


def test_button_edges_are_latched_even_if_main_thread_is_busy():
    buttons = iter([(True, False), (False, False), (True, False), (True, True), (False, False)])
    reader = pico.LatestBodyReader(SimpleNamespace(buttons=lambda: next(buttons)))
    for _ in range(5):
        reader.poll_controls()
    assert reader.has_controls()
    assert reader.take_controls() == ['toggle', 'toggle', 'exit']
    assert not reader.has_controls()


def fake_source(*, timestamps=(1_000_000_000, 1_000_000_000), poses=None):
    timestamps = iter(timestamps)
    if poses is None:
        poses = [[1, 2, 3, 0, 0, 0, 1]]
    sdk = SimpleNamespace(**{name: lambda: None for name in pico.SDK_GETTERS})
    sdk.get_body_timestamp_ns = lambda: next(timestamps)
    sdk.is_body_data_available = lambda: True
    sdk.get_body_joints_pose = lambda: poses
    sdk.get_A_button = lambda: True
    sdk.get_X_button = lambda: False
    sdk.get_time_stamp_ns = lambda: pytest.fail("Global SDK timestamp must never be used")
    streamer = SimpleNamespace(body_joint_names=["Pelvis"], coordinate_transform_unity_data=lambda body: body)
    return pico.XRoboSource(sdk, streamer)


def test_raw_body_reads_body_clock_twice_and_converts_xyzw():
    source = fake_source()
    sequence, body = source.read_body()
    assert sequence == 1_000_000_000
    assert body["Pelvis"] == [[1, 2, 3], [1, 0, 0, 0]]
    assert source.buttons() == (True, False)


def test_ignore_sdk_timestamps_never_calls_clock_getters_or_body_clock(monkeypatch):
    source = fake_source()
    source.ignore_sdk_timestamps = True
    def forbidden(*arguments, **keywords):
        pytest.fail('SDK/BodyClock timestamps must not be used in bypass mode')
    source.sdk.get_body_timestamp_ns = forbidden
    source.sdk.get_body_joints_timestamp = forbidden
    source.sdk.get_time_stamp_ns = forbidden
    current = SimpleNamespace(seconds=10.0)
    monkeypatch.setattr(pico.time, 'monotonic', lambda: current.seconds)
    monkeypatch.setattr(pico.time, 'time', lambda: 100.0)
    reader = pico.LatestBodyReader(source)
    reader.clock.map = forbidden
    reader.poll_once()
    first, generation, error = reader.snapshot()
    assert error is None and generation == 0
    assert first.source_time == 100
    current.seconds += .01
    reader.poll_once()
    assert reader.snapshot()[0] is first  # sampling is bounded, not 500 Hz cached-pose traffic
    current.seconds += .02
    reader.poll_once()
    second = reader.snapshot()[0]
    assert second.source_sequence > first.source_sequence
    assert second.source_time == pytest.approx(100.03)
    assert source.diagnostics()['acquisition_freshness_verified'] is False
    source.sdk.is_body_data_available = lambda: False
    current.seconds += .8
    reader.poll_once()
    assert reader.snapshot()[0] is second
    assert current.seconds - second.received_monotonic > .5


def test_ignore_sdk_timestamps_still_rejects_invalid_pose():
    source = fake_source(poses=[[0, 0, 0, 0, 0, 0, 0]])
    source.ignore_sdk_timestamps = True
    with pytest.raises(pico.SourceDiscontinuity, match='quaternions'):
        source.read_body()


def test_torn_snapshot_dropped_and_buttons_work_without_body():
    source = fake_source(timestamps=(1_000_000_000, 1_020_000_000))
    assert source.read_body() is None
    source.sdk.is_body_data_available = lambda: False
    assert source.read_body() is None
    assert source.buttons() == (True, False)


@pytest.mark.parametrize("poses", [[[0, 0, 0, 0, 0, 0, 0]], [[np.nan, 0, 0, 0, 0, 0, 1]],
                                  [[1001, 0, 0, 0, 0, 0, 1]], [[0, 0, 0]]])
def test_invalid_sdk_body_rejected(poses):
    with pytest.raises(pico.SourceDiscontinuity):
        fake_source(poses=poses).read_body()


def test_reader_latest_mailbox_static_pose_progress_and_stale_duplicates(monkeypatch):
    times = iter([1, 1.02, 1.4])
    sequences = iter([1_000_000_000, 1_020_000_000, 1_020_000_000])
    source = SimpleNamespace(read_body=lambda: (next(sequences), {"static": True}))
    reader = pico.LatestBodyReader(source)
    monkeypatch.setattr(pico.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(pico.time, "time", lambda: 100)
    reader.poll_once()
    reader.poll_once()
    latest, generation, error = reader.snapshot()
    assert latest.source_sequence == 1_020_000_000
    assert latest.received_monotonic == 1.02
    assert latest.source_time == pytest.approx(100.02)
    assert generation == 0 and error is None
    reader.poll_once()
    assert reader.snapshot()[0] is latest


def test_reader_discontinuity_clears_mailbox_and_increments_generation():
    sequences = iter([1_000_000_000, 900_000_000, 900_000_000])
    reader = pico.LatestBodyReader(SimpleNamespace(read_body=lambda: (next(sequences), {})))
    reader.poll_once()
    reader.poll_once()
    latest, generation, error = reader.snapshot()
    assert latest is None and generation == 1 and "regressed" in error
    reader.poll_once()
    assert reader.snapshot()[0].source_sequence == 900_000_000


def test_missing_body_timestamp_does_not_repeatedly_disarm_or_create_fresh_sample():
    reader = pico.LatestBodyReader(SimpleNamespace(read_body=lambda: (0, {})))
    for _ in range(5):
        reader.poll_once()
    latest, generation, error = reader.snapshot()
    assert latest is None
    assert generation == 1
    assert 'got 0 (int)' in error


def test_sdk_diagnostics_report_clocks_but_never_substitute_global_timestamp():
    source = fake_source(timestamps=(0, 0, 0))
    source.sdk.get_time_stamp_ns = lambda: 123456789
    source.sdk.get_body_joints_timestamp = lambda: [100, 200]
    report = source.diagnostics()
    assert report['get_body_timestamp_ns'] == 0
    assert report['get_time_stamp_ns'] == 123456789
    assert report['joint_timestamp_range'] == [100, 200]
    assert source.clock_source is None  # diagnostics must never select a clock
    del source.sdk.get_body_joints_timestamp
    with pytest.raises(pico.SourceDiscontinuity, match='unavailable'):
        source.read_body()  # global timestamp cannot make this executable


def joint_clock_source(timestamps):
    poses = [[0, 0, 1, 0, 0, 0, 1]] * 3
    source = fake_source(timestamps=(0,), poses=poses)
    source.streamer.body_joint_names = ['Pelvis', 'Left_Hip', 'Right_Hip']
    source.sdk.get_body_joints_timestamp = lambda: timestamps
    return source


def test_joint_body_clock_fallback_pins_one_index_and_ignores_global_clock():
    timestamps = [0, 1_000_000_000, 990_000_000]
    source = joint_clock_source(timestamps)
    sequence, body = source.read_body()
    assert sequence == 1_000_000_000
    assert source.clock_source == 'body_joint'
    assert source.clock_joint_index == 1
    assert body['Pelvis'][1] == [1, 0, 0, 0]
    # A different joint becoming newer must NOT switch the chosen clock.
    timestamps[2] = 5_000_000_000
    timestamps[1] += 20_000_000
    assert source.read_body()[0] == 1_020_000_000
    # A subsequently populated aggregate is another domain, not a replacement.
    source.sdk.get_body_timestamp_ns = lambda: 1793089264315686144
    assert source.read_body()[0] == 1_020_000_000


def test_joint_clock_zero_then_valid_can_start_but_selected_clock_loss_does_not_switch():
    timestamps = [0, 0, 0]
    source = joint_clock_source(timestamps)
    source.sdk.get_body_timestamp_ns = lambda: 0
    with pytest.raises(pico.SourceDiscontinuity, match='no positive'):
        source.read_body()
    assert source.clock_source is None
    timestamps[2] = 2_000_000_000
    assert source.read_body()[0] == 2_000_000_000
    timestamps[2] = 0
    timestamps[1] = 3_000_000_000
    with pytest.raises(pico.SourceDiscontinuity, match='became zero'):
        source.read_body()
    assert source.clock_joint_index == 2


def test_joint_clock_static_pose_new_samples_advance_but_frozen_clock_stales(monkeypatch):
    timestamps = [0, 1_000_000_000, 0]
    source = joint_clock_source(timestamps)
    reader = pico.LatestBodyReader(source)
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(pico.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(pico.time, 'time', lambda: 100 + clock.now)
    reader.poll_once()
    first, _, _ = reader.snapshot()
    timestamps[1] += 20_000_000
    clock.now += .02
    reader.poll_once()
    second, _, error = reader.snapshot()
    assert error is None
    assert second.source_time == pytest.approx(first.source_time + .02)
    assert second.body == first.body  # stationary human is still a fresh sample
    clock.now += .7
    reader.poll_once()
    assert reader.snapshot()[0] is second
    assert clock.now - second.received_monotonic > .5
    timestamps[1] -= 10_000_000
    reader.poll_once()
    latest, generation, error = reader.snapshot()
    assert latest is None and generation == 1 and 'regressed' in error


def test_joint_timestamp_snapshot_change_drops_sample_even_if_selected_index_unchanged():
    source = joint_clock_source([0, 1_000_000_000, 0])
    snapshots = iter([[0, 1_000_000_000, 0], [0, 1_000_000_000, 123]])
    source.sdk.get_body_joints_timestamp = lambda: next(snapshots)
    assert source.read_body() is None


@pytest.mark.parametrize('timestamps', [[0, 0], [0., 1., 2.], [0, -1, 10], [0, True, 'bad']])
def test_invalid_joint_clock_arrays_are_rejected(timestamps):
    with pytest.raises(pico.SourceDiscontinuity, match='joint timestamps'):
        joint_clock_source(timestamps).read_body()


def test_aggregate_clock_cannot_switch_to_joint_clock_when_it_later_disappears():
    source = fake_source(timestamps=(1_000_000_000, 1_000_000_000, 0, 0))
    source.sdk.get_body_joints_timestamp = lambda: [2_000_000_000]
    assert source.read_body()[0] == 1_000_000_000
    assert source.read_body()[0] == 0  # BodyClock rejects this, never switches domains
    assert source.clock_source == 'body'


def test_reader_shutdown_joins_and_closes_sdk():
    closed = []
    source = SimpleNamespace(read_body=lambda: None, close=lambda: closed.append(True))
    reader = pico.LatestBodyReader(source)
    reader.start()
    reader.close()
    assert not reader._thread.is_alive()
    assert closed == [True]


def test_new_arm_discards_existing_sample_waits_for_chunk_then_acks_start(session):
    step(session, 1_000_000_000)
    assert not session.buffer.samples
    start_session(session)
    assert session.publisher.starts[0].sequence == 0
    assert session.publisher.starts[0].state == "start"
    assert session.publisher.starts[0].source_sequence == 1_040_000_000
    step(session, 1_080_000_000, now=1.08)
    assert len(session.publisher.starts) == 1
    assert session.publisher.frames[0].state == "frame"
    assert session.publisher.frames[0].sequence == 1
    assert session.publisher.frames[0].source_time == pytest.approx(101.08)
    assert session.publisher.frames[0].reference_time == pytest.approx(100.98)


def test_exit_precedes_a_and_cleanup_stop_has_no_reference(session):
    start_session(session)
    assert not step(session, 1_060_000_000, right_a=True, left_x=True, now=1.06)
    assert not session.armed
    assert session.publisher.frames[-1].state == "stop"
    assert session.publisher.frames[-1].reference is None
    assert not session.buffer.samples


def test_pause_edge_requires_new_edge_and_new_full_buffer(session):
    start_session(session)
    first_session = session.session_id
    step(session, 1_060_000_000, right_a=True, now=1.06)
    step(session, 1_080_000_000, right_a=True, now=1.08)
    assert not session.armed and not session.buffer.samples
    step(session, 1_100_000_000, right_a=False, now=1.1)
    step(session, 1_120_000_000, right_a=True, now=1.12)
    assert session.armed and not session.buffer.samples
    step(session, 1_140_000_000, now=1.14)
    assert len(session.publisher.starts) == 1
    step(session, 1_180_000_000, now=1.18)
    assert session.session_id != first_session
    assert len(session.publisher.starts) == 2


def test_operator_press_during_ik_cancels_before_publishing_another_target(session):
    start_session(session)
    frame_count = len(session.publisher.frames)
    def interrupted_ik(body):
        session.cancel_requested = lambda: True
        return np.zeros(36)
    session.retargeter.retarget = interrupted_ik
    step(session, 1_060_000_000, now=1.06)
    assert not session.armed
    assert session.operator_cancelled
    assert len(session.publisher.frames) == frame_count + 1
    assert session.publisher.frames[-1].state == 'stop'


def test_stale_body_stops_even_if_controller_and_global_clock_are_fresh(session):
    start_session(session)
    frozen = sample(1_040_000_000, 1.04)
    session.step(right_a=False, left_x=False, sample=frozen, generation=0,
                 status=status(session), now=1.55)
    assert not session.armed and not session.buffer.samples
    step(session, 1_600_000_000, now=1.6)
    assert len(session.publisher.starts) == 1


@pytest.mark.parametrize("updates", [{"paused": True}, {"state": "stopped"}, {"state": "stale"},
                                     {"state": "error"}, {"stream_mode": "recorded"},
                                     {"controller_id": "replacement"}])
def test_receiver_interruption_clears_and_does_not_auto_rearm(session, updates):
    start_session(session)
    interrupted = status(session, **updates)
    step(session, 1_060_000_000, receiver=interrupted, now=1.06)
    assert not session.armed and not session.buffer.samples
    step(session, 1_080_000_000, now=1.08)
    assert not session.armed
    assert len(session.publisher.starts) == 1


def test_source_generation_change_disarms_even_if_a_pressed(session):
    start_session(session)
    step(session, 1_060_000_000, right_a=True, generation=1, now=1.06)
    assert not session.armed and not session.buffer.samples
    step(session, 1_080_000_000, right_a=True, generation=1, now=1.08)
    assert not session.armed


@pytest.mark.parametrize("receiver_state", ["buffering", "holding"])
def test_live_receiver_healthy_states_do_not_pause(session, receiver_state):
    start_session(session)
    step(session, 1_080_000_000, receiver=status(session, state=receiver_state), now=1.08)
    assert session.armed
    assert session.publisher.frames[-1].state == "frame"


def test_stopped_receiver_requires_explicit_edge_then_refills_new_buffer(session):
    start_session(session)
    stopped = status(session, state="stopped")
    step(session, 1_060_000_000, receiver=stopped, now=1.06)
    assert not session.armed
    step(session, 1_080_000_000, receiver=stopped, now=1.08)
    assert not session.armed and not session.buffer.samples
    step(session, 1_100_000_000, right_a=True, receiver=stopped, now=1.10)
    assert session.armed and not session.buffer.samples
    step(session, 1_120_000_000, receiver=stopped, now=1.12)
    step(session, 1_160_000_000, receiver=stopped, now=1.16)
    assert len(session.publisher.starts) == 2


def test_actual_overlapping_buffer_keeps_publishing_when_first_row_is_unchanged(session):
    from deployment.teleop_reference import BufferedReference

    session.buffer = BufferedReference(2)
    qpos = np.zeros(36)
    qpos[3] = 1
    session.retargeter.retarget = lambda body: qpos.copy()
    step(session, 1_000_000_000, right_a=True)
    for frame in range(1, 26):
        step(session, 1_000_000_000 + frame * 20_000_000, now=1 + frame * 0.02)
    packets = session.publisher.starts + session.publisher.frames
    assert len(packets) > 5
    assert all(packet.reference_time == packets[0].reference_time for packet in packets)
    assert len(packets[-1].reference) > len(packets[0].reference)
    assert all(packet.source_time > previous.source_time for previous, packet in zip(packets, packets[1:]))


def test_retarget_error_fails_closed_and_close_uses_owned_publisher(session):
    start_session(session)
    session.retargeter.retarget = lambda body: (_ for _ in ()).throw(ValueError("invalid quaternion"))
    step(session, 1_080_000_000, now=1.08)
    assert not session.armed
    assert session.publisher.frames[-1].state == "stop"
    session.close()
    assert session.publisher.closed == [None]


def test_interrupt_during_start_retains_stop_for_close(session):
    session.publisher.start = lambda packet: (_ for _ in ()).throw(KeyboardInterrupt())
    step(session, 1_000_000_000, right_a=True)
    step(session, 1_020_000_000, now=1.02)
    with pytest.raises(KeyboardInterrupt):
        step(session, 1_040_000_000, now=1.04)
    session.close()
    assert session.publisher.closed[0].state == "stop"
    assert session.publisher.closed[0].sequence == 1


def mock_gmr():
    names = [f"joint_{index}" for index in range(29)]
    model = SimpleNamespace(nq=36, jnt_type=np.array([0] + [3] * 29),
                            jnt_limited=np.array([False] + [True] * 29),
                            jnt_qposadr=np.array([0] + list(range(7, 36))),
                            jnt_range=np.array([[0, 0]] + [[-2, 2]] * 29))
    mujoco = SimpleNamespace(mjtJoint=SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3),
                             mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
                             mj_name2id=lambda model, kind, name: 29 - names.index(name))
    qpos = np.zeros(36)
    qpos[3] = 1
    qpos[7:] = np.arange(29) / 29
    retargeter = SimpleNamespace(model=model, retarget=lambda body, *, offset_to_ground: qpos.copy())
    return pico.GMRRetargeter(retargeter, mujoco, names), qpos


def test_gmr_maps_named_hinges_not_blind_qpos_slice():
    adapter, qpos = mock_gmr()
    result = adapter.retarget({})
    np.testing.assert_array_equal(result[7:], qpos[7:][::-1])
    np.testing.assert_array_equal(result[:7], qpos[:7])


@pytest.mark.parametrize("index,value", [(3, 0), (7, 3), (1, np.nan), (2, 1001)])
def test_gmr_rejects_bad_quaternion_limits_and_positions(index, value):
    adapter, qpos = mock_gmr()
    qpos[index] = value
    with pytest.raises(ValueError):
        adapter.retarget({})


def test_gmr_rejects_unlimited_or_missing_hinges():
    adapter, _ = mock_gmr()
    adapter.model.jnt_limited[1] = False
    names = [f"joint_{index}" for index in range(29)]
    mujoco = SimpleNamespace(mjtJoint=SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3),
                             mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
                             mj_name2id=lambda model, kind, name: names.index(name) + 1)
    with pytest.raises(ValueError, match="limited hinge"):
        pico.GMRRetargeter(adapter.retargeter, mujoco, names)


def simulator_parameters(names, *, upper=2.0):
    return {"joint_names": list(names), "joint_lower": [-2.0] * 29, "joint_upper": [upper] * 29}


def test_simulator_limits_map_by_name_and_reject_pose_inside_only_gmr_limits():
    adapter, _ = mock_gmr()
    parameters = simulator_parameters(reversed(adapter.joint_names))
    parameters["joint_upper"][-1] = 0.5  # joint_0 is last in this simulator order.
    adapter.configure_simulator_limits(parameters)
    assert adapter.simulator_limits[0, 1] == 0.5
    assert adapter.simulator_limits[1, 1] == 2.0
    with pytest.raises(ValueError, match="simulator MJCF joint limits: joint_0"):
        adapter.retarget({})


def test_simulator_limits_accept_in_range_pose_without_modifying_it():
    adapter, _ = mock_gmr()
    expected = adapter.retarget({})
    adapter.configure_simulator_limits(simulator_parameters(adapter.joint_names, upper=1.0))
    np.testing.assert_array_equal(adapter.retarget({}), expected)


@pytest.mark.parametrize("field,value", [
    ("joint_names", ["missing"] * 29), ("joint_lower", [0] * 28),
    ("joint_upper", [np.nan] * 29), ("joint_upper", [-3] * 29),
    ("joint_lower", [3] * 29),
])
def test_simulator_limits_reject_invalid_named_ranges(field, value):
    adapter, _ = mock_gmr()
    parameters = simulator_parameters(adapter.joint_names)
    parameters[field] = value
    with pytest.raises(ValueError, match="Simulator limits"):
        adapter.configure_simulator_limits(parameters)
    assert adapter.simulator_limits is None


def test_simulator_limits_reject_disjoint_gmr_ranges():
    adapter, _ = mock_gmr()
    parameters = simulator_parameters(adapter.joint_names, upper=4)
    parameters["joint_lower"] = [3] * 29
    with pytest.raises(ValueError, match="no common valid interval"):
        adapter.configure_simulator_limits(parameters)


@pytest.mark.parametrize("failure_phase", ["limits", "acquire", "wait_ready", "mode"])
def test_run_initializes_before_lease_and_closes_sdk_on_startup_failure(monkeypatch, failure_phase):
    from deployment import contract, live_reference

    events = []

    class StartupPublisher:
        def __init__(self, client, channel, timeout):
            pass

        def acquire(self):
            events.append("acquire")
            assert "load_adapters" in events and "configure_limits" in events
            if failure_phase == "acquire":
                raise RuntimeError("acquire failed")

        def wait_ready(self):
            events.append("wait_ready")
            if failure_phase == "wait_ready":
                raise TimeoutError("wait failed")
            return {"stream_mode": "recorded"}

        def close(self):
            events.append("publisher.close")

    def configure_limits(parameters):
        events.append("configure_limits")
        assert parameters == {"compiled": True}
        if failure_phase == "limits":
            raise ValueError("limits failed")

    def load_adapters(root, height):
        events.append("load_adapters")
        assert "acquire" not in events
        source = SimpleNamespace(close=lambda: events.append("source.close"))
        return source, SimpleNamespace(configure_simulator_limits=configure_limits)

    monkeypatch.setattr(contract, "robot_parameters", lambda path: {"compiled": True})
    monkeypatch.setattr(live_reference, "LivePublisher", StartupPublisher)
    monkeypatch.setattr(pico, "load_adapters", load_adapters)
    monkeypatch.setattr(pico, "connect_redis", lambda *args: SimpleNamespace(
        close=lambda: events.append("client.close")))
    args = SimpleNamespace(redis_host="127.0.0.1", redis_port=6379, redis_db=0,
                           channel="test", timeout=1, wbc_root="external", human_height=1.6,
                           mjcf="simulator.xml")
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        pico.run(args)
    assert events.count("source.close") == 1
    assert events[-3:] == ["publisher.close", "source.close", "client.close"]


def test_check_never_imports_sdk_or_gmr_or_changes_import_path(tmp_path, monkeypatch):
    package = tmp_path / "data_processing" / "general_motion_retargeting"
    package.mkdir(parents=True)
    for path in (package / "xrobot_utils.py", package / "motion_retarget.py",
                 package / "ik_configs" / "xrobot_to_g1.json",
                 package.parent / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test provenance")
    original_import = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("xrobotoolkit_sdk", "general_motion_retargeting", "mujoco"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    monkeypatch.setattr(pico.importlib.util, "find_spec", lambda name: None)
    report = pico.check_environment(tmp_path)
    assert not report["hardware_initialized"]
    assert "get_body_timestamp_ns" in report["sdk_required_getters"]
    assert report["external"]["model"]["sha256"]
    assert report["external"]["simulator_mjcf"]["path"] == str(pico.DEFAULT_MJCF)
    assert "without compiling" in report["simulator_limit_validation"]
    assert not report["modules"]["daqp"]["available"]
    assert str(package.parent) not in pico.sys.path


def test_cli_requires_external_checkout_and_explicit_mode():
    with pytest.raises(SystemExit):
        pico.main(["--check"])
    with pytest.raises(SystemExit):
        pico.main(["--wbc-root", "/unused"])
