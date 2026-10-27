"""SDK-free checks for measured teleoperation references and cursor overlap."""

import math

import numpy as np
import pytest

from deployment.teleop_reference import BufferedReference, CONTROL_DT


UNIX_START = 1_790_000_000.0
ROOT_VELOCITY = np.array([0.7, -0.3, 0.2])
JOINT_VELOCITY = np.linspace(-0.8, 1.2, 29)


def analytic_qpos(elapsed, *, yaw=0.0, pitch_rate=0.8):
    """Yaw followed by moving pitch: world angular axis differs from body axis."""
    half_pitch = (0.3 + pitch_rate * elapsed) / 2
    half_yaw = yaw / 2
    result = np.empty(36)
    result[:3] = np.array([4.0, -3.0, 0.91]) + ROOT_VELOCITY * elapsed
    result[3:7] = [
        math.cos(half_yaw) * math.cos(half_pitch),
        -math.sin(half_yaw) * math.sin(half_pitch),
        math.cos(half_yaw) * math.sin(half_pitch),
        math.sin(half_yaw) * math.cos(half_pitch),
    ]
    result[7:] = np.linspace(-0.2, 0.4, 29) + JOINT_VELOCITY * elapsed
    return result


def feed(buffer, times, *, first_sequence=0, yaw=0.0, sign_flips=False):
    for sequence, elapsed in enumerate(times, start=first_sequence):
        position = analytic_qpos(elapsed, yaw=yaw)
        if sign_flips and sequence % 2:
            position[3:7] *= -3.0
        assert buffer.add(position, source_sequence=sequence, source_time=UNIX_START + elapsed)


def test_nonuniform_samples_preserve_linear_motion_and_world_rotation():
    buffer = BufferedReference(5, filter_tau=0, recenter=False)
    times = [0.0, 0.011, 0.039, 0.057, 0.088, 0.103, 0.139, 0.157, 0.181, 0.213, 0.241]
    feed(buffer, times, yaw=math.pi / 2, sign_flips=True)
    reference_time, rows = buffer.chunk()
    elapsed = reference_time - UNIX_START + np.arange(len(rows)) * CONTROL_DT
    expected = np.stack([analytic_qpos(timestamp, yaw=math.pi / 2) for timestamp in elapsed])
    assert rows.shape[1] == 71
    np.testing.assert_allclose(rows[:, :36], expected, atol=2e-7)
    np.testing.assert_allclose(rows[:, 36:65], np.broadcast_to(JOINT_VELOCITY, (len(rows), 29)), atol=6e-6)
    np.testing.assert_allclose(rows[:, 65:68], np.broadcast_to(ROOT_VELOCITY, (len(rows), 3)), atol=4e-6)
    np.testing.assert_allclose(rows[:, 68:71], np.broadcast_to([-0.8, 0, 0], (len(rows), 3)), atol=4e-6)
    np.testing.assert_allclose(np.linalg.norm(rows[:, 3:7], axis=1), 1, atol=1e-12)
    assert np.all(np.sum(rows[1:, 3:7] * rows[:-1, 3:7], axis=1) > 0)
    assert np.isfinite(rows).all()
    assert reference_time <= buffer.source_time - buffer.startup_delay_seconds + 1e-6
    assert buffer.source_time - buffer.startup_delay_seconds - reference_time < CONTROL_DT + 1e-6
    assert reference_time + len(rows) * CONTROL_DT <= buffer.source_time + 1e-6


def test_startup_waits_for_horizon_jitter_and_derivative_then_keeps_overlap():
    buffer = BufferedReference(5, filter_tau=0, recenter=False)
    assert buffer.chunk() is None
    feed(buffer, [0.0, 0.04, 0.08, 0.12, 0.159])
    assert buffer.chunk() is None
    feed(buffer, [0.16], first_sequence=5)
    first_time, first_rows = buffer.chunk()
    assert first_time == UNIX_START
    assert len(first_rows) >= buffer.goal_frames + 1
    assert buffer.delay_seconds == pytest.approx(0.16, abs=1e-6)
    feed(buffer, [0.20, 0.25, 0.31, 0.37, 0.44, 0.5, 0.56, 0.62], first_sequence=6)
    next_time, next_rows = buffer.chunk()
    assert next_time == first_time
    np.testing.assert_array_equal(next_rows[:len(first_rows)], first_rows)
    assert len(next_rows) > len(first_rows)
    # The receiver's old delayed cursor, not just a fresh K+1 window, survives.
    assert next_time <= buffer.source_time - 0.5


def test_first_chunk_after_long_capture_is_not_delayed_by_entire_history():
    buffer = BufferedReference(10, filter_tau=0)
    feed(buffer, np.arange(201) * 0.01)
    reference_time, rows = buffer.chunk()
    assert buffer.source_time - reference_time == pytest.approx(0.26, abs=1e-6)
    assert len(rows) == 13
    assert reference_time + (len(rows) - 1) * CONTROL_DT <= buffer.source_time - CONTROL_DT + 1e-6


def test_centered_gradients_use_a_real_next_sample_even_at_last_emitted_row():
    buffer = BufferedReference(2, filter_tau=0, recenter=False, jitter_margin=0)
    for sequence in range(21):
        elapsed = sequence * CONTROL_DT
        position = analytic_qpos(elapsed)
        position[0] = elapsed**2
        position[7] = 3 * elapsed**2
        buffer.add(position, source_sequence=sequence, source_time=100 + elapsed)
    reference_time, rows = buffer.chunk()
    elapsed = reference_time - 100 + np.arange(len(rows)) * CONTROL_DT
    np.testing.assert_allclose(rows[:, 65], 2 * elapsed, atol=1e-12)
    np.testing.assert_allclose(rows[:, 36], 6 * elapsed, atol=1e-12)
    assert elapsed[-1] == pytest.approx(0.38)
    assert rows[-1, 65] == pytest.approx(0.76)
    # Repeated reads neither extrapolate nor append synthetic hold frames.
    repeated_time, repeated_rows = buffer.chunk()
    assert repeated_time == reference_time
    np.testing.assert_array_equal(repeated_rows, rows)


def test_fixed_recenter_rotates_positions_quaternions_and_world_velocities():
    buffer = BufferedReference(3, filter_tau=0, recenter=True)
    feed(buffer, np.arange(31) * 0.01, yaw=math.pi / 2)
    reference_time, rows = buffer.chunk()
    elapsed = reference_time - UNIX_START + np.arange(len(rows)) * CONTROL_DT
    expected_root = elapsed[:, None] * np.array([-0.3, -0.7, 0.2])
    expected_root[:, 2] += 0.91
    np.testing.assert_allclose(rows[:, :3], expected_root, atol=2e-7)
    expected_quaternion = np.stack([analytic_qpos(timestamp)[3:7] for timestamp in elapsed])
    np.testing.assert_allclose(rows[:, 3:7], expected_quaternion, atol=2e-7)
    np.testing.assert_allclose(rows[:, 65:68], np.broadcast_to([-0.3, -0.7, 0.2], (len(rows), 3)), atol=4e-6)
    np.testing.assert_allclose(rows[:, 68:71], np.broadcast_to([0, 0.8, 0], (len(rows), 3)), atol=4e-6)
    assert rows[0, 2] > 0.9  # No arbitrary floor/height shift.


def test_dt_dependent_filter_matches_exponential_step_for_positions_and_rotation():
    buffer = BufferedReference(2, filter_tau=0.07, jitter_margin=0, recenter=False)
    initial = analytic_qpos(0, pitch_rate=0)
    initial[0] = 0
    initial[7:] = 0
    buffer.add(initial, source_sequence=0, source_time=100)
    times = [0.013, 0.044, 0.071, 0.105, 0.139, 0.18, 0.213]
    for sequence, elapsed in enumerate(times, start=1):
        target = analytic_qpos(1.0)
        target[0] = 1
        target[7:] = 2
        target[3:7] *= -1 if sequence % 2 else 1
        buffer.add(target, source_sequence=sequence, source_time=100 + elapsed)
    reference_time, rows = buffer.chunk()
    grid_times = reference_time - 100 + np.arange(len(rows)) * CONTROL_DT
    source_times = np.array([0, *times])
    response = 1 - np.exp(-source_times / buffer.filter_tau)
    interpolated_response = np.interp(grid_times, source_times, response)
    np.testing.assert_allclose(rows[:, 0], interpolated_response, atol=1e-12)
    np.testing.assert_allclose(rows[:, 7:36], np.broadcast_to(2 * interpolated_response[:, None], (len(rows), 29)), atol=1e-12)
    expected_pitch = 0.3 + 0.8 * interpolated_response
    np.testing.assert_allclose(rows[:, 3], np.cos(expected_pitch / 2), atol=1e-12)
    np.testing.assert_allclose(rows[:, 5], np.sin(expected_pitch / 2), atol=1e-12)


def test_duplicate_regression_gaps_metadata_and_clear_rearm():
    buffer = BufferedReference(2)
    position = analytic_qpos(0)
    original = position.copy()
    assert buffer.add(position, source_sequence=12, source_time=100)
    np.testing.assert_array_equal(position, original)
    assert not buffer.add(position, source_sequence=12, source_time=100)
    assert buffer.diagnostics["duplicates"] == 1
    for sequence, timestamp in [(11, 100.01), (12, 100.01), (13, 100), (13, 99.99), (13, 100.151)]:
        with pytest.raises(ValueError, match="regression|gap"):
            buffer.add(position, source_sequence=sequence, source_time=timestamp)
        assert buffer.source_sequence == 12
        assert buffer.source_time == 100
    assert buffer.add(position, source_sequence=20, source_time=100.15)
    assert buffer.source_sequence == 20  # Dropped sequence IDs are allowed.
    buffer.chunk()
    buffer.clear()
    assert buffer.source_sequence is None and buffer.source_time is None
    assert buffer.delay_seconds is None and buffer.chunk() is None
    assert buffer.diagnostics["raw_samples"] == 0
    assert buffer.add(position, source_sequence=0, source_time=50)
    assert buffer.chunk() is None


def test_history_and_raw_storage_remain_bounded_during_long_stream():
    buffer = BufferedReference(5, filter_tau=0, recenter=False)
    previous_chunk = None
    for sequence in range(1201):
        elapsed = sequence * 0.01
        buffer.add(analytic_qpos(elapsed), source_sequence=sequence, source_time=UNIX_START + elapsed)
        if sequence % 11:
            continue
        chunk = buffer.chunk()
        if chunk is None:
            continue
        reference_time, rows = chunk
        assert (len(rows) - 1) * CONTROL_DT <= buffer.history_seconds + 1e-9
        assert reference_time <= buffer.source_time - 0.5 or elapsed < 0.6
        assert buffer.diagnostics["raw_duration_seconds"] <= buffer.diagnostics["raw_retention_limit_seconds"] + 1e-6
        assert buffer.diagnostics["raw_samples"] <= 175
        if previous_chunk is not None:
            previous_time, previous_rows = previous_chunk
            offset = round((reference_time - previous_time) / CONTROL_DT)
            overlap = len(previous_rows) - offset
            np.testing.assert_allclose(rows[:overlap], previous_rows[offset:], rtol=0, atol=1e-12)
        previous_chunk = chunk
    assert buffer.source_sequence == 1200
    assert buffer.diagnostics["accepted_samples"] == 1201
    assert buffer.diagnostics["raw_samples"] < 200


def test_invalid_inputs_do_not_seed_or_mutate_buffer():
    buffer = BufferedReference(2)
    for position in [np.zeros(35), np.zeros(36), np.full(36, np.nan), np.full(36, np.inf), np.full(36, "bad")]:
        with pytest.raises(ValueError):
            buffer.add(position, source_sequence=0, source_time=100)
        assert buffer.source_time is None
    for sequence, timestamp in [(-1, 100), (True, 100), (1.5, 100), (0, np.nan), (0, np.inf), (0, "100")]:
        with pytest.raises(ValueError):
            buffer.add(analytic_qpos(0), source_sequence=sequence, source_time=timestamp)
        assert buffer.source_time is None
    for options in [{"filter_tau": -1}, {"jitter_margin": -1}, {"max_gap": 0}, {"history_seconds": 0.01}, {"filter_tau": np.nan}]:
        with pytest.raises(ValueError):
            BufferedReference(2, **options)
    with pytest.raises(ValueError):
        BufferedReference(True)
