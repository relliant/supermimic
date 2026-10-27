"""Standalone ONNX control timing/reset tests; no DDS or robot connections."""

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from deployment import sim2sim
from deployment.contract import DEFAULT_JOINT_POSITION, G1_JOINT_NAMES, build_goals
from deployment.reference_receiver import ReferenceReceiver
from deployment.reference_stream import ReferencePacket


class RecordingPolicy:
    history_length = 3
    goal_frames = 2

    def __init__(self, path):
        self.path = Path(path)
        self.calls = []

    def infer(self, history, goal):
        self.calls.append((history.copy(), goal.copy()))
        return np.full(29, 0.25, dtype=np.float32)


@pytest.fixture
def simulation(tmp_path):
    joint_bodies = ''.join(
        f'<body name="body_{index}" pos="0 0 {0.01 * index}">'
        f'<joint name="{joint_name}" type="hinge" limited="true" range="-3 3"/>'
        '<geom type="sphere" size="0.01" mass="0.01"/></body>'
        for index, joint_name in reversed(list(enumerate(G1_JOINT_NAMES)))
    )
    model_path = tmp_path / 'robot.xml'
    model_path.write_text(
        '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis" pos="0 0 1">'
        '<freejoint/><geom type="sphere" size="0.1" mass="10"/>'
        '<geom name="visual_only" type="sphere" size="0.2" contype="0" conaffinity="0" mass="0"/>'
        + joint_bodies + '</body></worldbody></mujoco>'
    )
    policy_path = tmp_path / 'policy.onnx'
    policy_path.write_bytes(b'fake policy for controlled stepping, no inference file loaded')
    motion_path = tmp_path / 'reference.csv'
    motion_path.write_text('fixture provenance')
    reference = np.zeros((6, 71), dtype=np.float32)
    reference[:, 2] = 1.0
    reference[:, 3] = 1.0
    reference[:, 7:36] = DEFAULT_JOINT_POSITION
    reference[:, 65] = 0.1
    clip = sim2sim.MotionClip(motion_path, reference, build_goals(reference, 2))
    return sim2sim.PolicySimulation(RecordingPolicy(policy_path), model_path, clip)


def test_control_timing_history_and_applied_action(simulation):
    simulation.advance()
    history, goal = simulation.policy.calls[0]
    np.testing.assert_array_equal(history[0], history[-1])
    np.testing.assert_array_equal(goal, simulation.clip.goals[0])
    np.testing.assert_allclose(simulation.target, simulation.physics.default_joint_position + 0.25 * simulation.physics.action_scale)
    assert simulation.data.time == pytest.approx(0.02)
    assert simulation.frame == 1
    first_observation = history[-1].copy()
    simulation.advance()
    history, goal = simulation.policy.calls[1]
    np.testing.assert_array_equal(history[-2], first_observation)
    # Previous action occupies features 64..92 in the unscaled 99D contract.
    np.testing.assert_array_equal(history[-1, 64:93], np.full(29, 0.25))
    np.testing.assert_array_equal(goal, simulation.clip.goals[1])
    assert simulation.data.time == pytest.approx(0.04)
    assert simulation.sample_count == 2


def test_reset_changes_reference_and_resets_controller_transaction(simulation, tmp_path):
    simulation.advance()
    reference = simulation.clip.reference.copy()
    reference[:, 0] += 2
    reference[:, 7] += 0.1
    other_path = tmp_path / 'other.csv'
    other_path.write_text('another clip')
    simulation.reset(sim2sim.MotionClip(other_path, reference, build_goals(reference, 2)))
    assert simulation.frame == 0
    assert simulation.data.time == 0
    assert simulation.history is None
    assert simulation.sample_count == 0
    np.testing.assert_array_equal(simulation.previous_action, np.zeros(29))
    np.testing.assert_allclose(simulation.data.qpos[simulation.mapping.qpos_addresses], reference[0, 7:36])
    np.testing.assert_allclose(simulation.data.qvel[simulation.mapping.root_dof_address], 0.1)
    simulation.advance()
    np.testing.assert_array_equal(simulation.policy.calls[-1][1], simulation.clip.goals[0])


def test_ghost_does_not_modify_physics_or_add_collisions(simulation):
    import mujoco

    simulation.advance()
    actual_position = simulation.data.qpos.copy()
    actual_velocity = simulation.data.qvel.copy()
    geometry_count = simulation.model.ngeom
    simulation.update_ghost(anchor_xy=True)
    scene = mujoco.MjvScene(simulation.model, maxgeom=200)
    simulation.add_ghost(scene, mujoco.MjvOption())
    assert scene.ngeom > 0
    assert scene.ngeom <= geometry_count
    assert simulation.model.ngeom == geometry_count
    visual_id = simulation.model.geom('visual_only').id
    assert simulation.model.geom_contype[visual_id] == 0
    assert simulation.model.geom_conaffinity[visual_id] == 0
    assert scene.geoms[0].rgba[3] == pytest.approx(0.3)
    np.testing.assert_array_equal(simulation.data.qpos, actual_position)
    np.testing.assert_array_equal(simulation.data.qvel, actual_velocity)


def test_goal_tail_keeps_recorded_velocity_but_zeros_synthetic_hold(simulation):
    reference = simulation.clip.reference.copy()
    reference[:, 36:65] = 2.0
    reference[:, 65:71] = 3.0
    goals = build_goals(reference, 2).reshape(len(reference), 2, 73)
    legacy_goals = build_goals(reference, 2, hold_final_velocity=True).reshape(len(reference), 2, 73)
    np.testing.assert_array_equal(goals[-2, 0], legacy_goals[-2, 0])
    np.testing.assert_array_equal(goals[-2, 1, 9:15], np.zeros(6))
    np.testing.assert_array_equal(goals[-2, 1, 44:], np.zeros(29))
    np.testing.assert_array_equal(legacy_goals[-2, 1, 9:15], np.full(6, 3.0))
    np.testing.assert_allclose(legacy_goals[-2, 1, 44:], 0.2)


def test_end_of_clip_does_not_silently_loop(simulation):
    for _ in range(5):
        simulation.advance()
    assert simulation.finished
    assert simulation.data.time == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match='Clip ended'):
        simulation.advance()


def test_stream_starts_without_clip_then_waits_for_valid_session(simulation):
    waiting = sim2sim.PolicySimulation(simulation.policy, simulation.mjcf)
    receiver = ReferenceReceiver('controller', 2, 0.5)
    assert waiting.data.time == 0
    assert waiting.history is None
    assert receiver.accept(None, waiting, wall_time=100, monotonic_time=0) is False
    reference = simulation.clip.reference[:3]
    packet = ReferencePacket('controller', 'session', 0, 100, 'start', 'remote_walk', 'recorded', reference)
    assert receiver.accept(packet.encode(), waiting, wall_time=100, monotonic_time=0)
    assert waiting.data.time == pytest.approx(0.02)
    assert waiting.clip.name == 'remote_walk'
    assert receiver.status(paused=False)['sequence'] == 0
    assert not receiver.accept(packet.encode(), waiting, wall_time=100.1, monotonic_time=0.1)
    assert waiting.data.time == pytest.approx(0.02)


def test_stream_frames_preserve_physics_and_history_and_stop_does_not_step(simulation):
    receiver = ReferenceReceiver('controller', 2, 0.5)
    reference = simulation.clip.reference[:3].copy()
    packet = ReferencePacket('controller', 'session', 0, 100, 'start', 'remote', 'recorded', reference)
    assert receiver.accept(packet.encode(), simulation, wall_time=100, monotonic_time=0)
    original_history = simulation.history.copy()
    reference[:, 0] = 100  # A new reference root must not teleport the physical root.
    update = replace(packet, state='frame', sequence=1, reference=reference)
    assert receiver.accept(update.encode(), simulation, wall_time=100.02, monotonic_time=0.02)
    assert simulation.data.time == pytest.approx(0.04)
    assert simulation.data.qpos[0] < 1
    np.testing.assert_array_equal(simulation.history[-2], original_history[-1])
    stop = replace(packet, state='stop', sequence=2, reference=None)
    assert not receiver.accept(stop.encode(), simulation, wall_time=100.04, monotonic_time=0.04)
    assert receiver.state == 'stopped'
    assert receiver.sequence == 2
    assert simulation.data.time == pytest.approx(0.04)
    assert not receiver.accept(replace(update, sequence=3).encode(), simulation, wall_time=100.05, monotonic_time=0.05)
    assert simulation.data.time == pytest.approx(0.04)


def test_stream_timeout_rejects_replay_and_requires_new_session(simulation):
    receiver = ReferenceReceiver('controller', 2, 0.5)
    packet = ReferencePacket('controller', 'session', 0, 100, 'start', 'walk', 'recorded', simulation.clip.reference[:3])
    receiver.accept(packet.encode(), simulation, wall_time=100, monotonic_time=0)
    receiver.accept(None, simulation, wall_time=100.6, monotonic_time=0.6)
    assert receiver.state == 'stale'
    replay = replace(packet, state='frame', sequence=1, sent_at=100.6)
    assert not receiver.accept(replay.encode(), simulation, wall_time=100.6, monotonic_time=0.6)
    assert simulation.data.time == pytest.approx(0.02)
    restart = replace(packet, session_id='new_session', sent_at=100.6)
    assert receiver.accept(restart.encode(), simulation, wall_time=100.6, monotonic_time=0.6)
    assert receiver.state == 'playing'
    assert simulation.data.time == pytest.approx(0.02)  # explicit simulation reset


def test_stream_invalid_or_wrong_controller_cannot_run_physics(simulation):
    receiver = ReferenceReceiver('controller', 2, 0.5)
    packet = ReferencePacket('old_controller', 'session', 0, 100, 'start', 'walk', 'recorded', simulation.clip.reference[:3])
    assert not receiver.accept(packet.encode(), simulation, wall_time=100, monotonic_time=0)
    assert simulation.data.time == 0
    assert receiver.session_id is None
    assert not receiver.accept(b'{"malformed":true}', simulation, wall_time=100, monotonic_time=0)
    assert receiver.state == 'error'
    assert simulation.data.time == 0


def test_fresh_packet_after_elapsed_idle_timeout_requires_new_start(simulation):
    receiver = ReferenceReceiver('controller', 2, 0.5)
    packet = ReferencePacket('controller', 'session', 0, 100, 'start', 'walk', 'recorded', simulation.clip.reference[:3])
    assert receiver.accept(packet.encode(), simulation, wall_time=100, monotonic_time=0)
    fresh_but_late = replace(packet, state='frame', sequence=1, sent_at=101)
    assert not receiver.accept(fresh_but_late.encode(), simulation, wall_time=101, monotonic_time=1)
    assert receiver.state == 'stale'
    assert simulation.data.time == pytest.approx(0.02)


def test_headless_cli_records_actual_state_and_exact_time(simulation, monkeypatch, tmp_path):
    monkeypatch.setattr(sim2sim, 'OnnxPolicy', lambda path: simulation.policy)
    monkeypatch.setattr(sim2sim.MotionClip, 'load', lambda *arguments: simulation.clip)
    trajectory_path = tmp_path / 'states.npz'
    metrics_path = tmp_path / 'metrics.json'
    assert sim2sim.main([
        '--policy', str(simulation.policy.path), '--motion', str(simulation.clip.path),
        '--mjcf', str(simulation.mjcf), '--headless', '--no-realtime', '--seconds', '0.06',
        '--record', str(trajectory_path), '--metrics', str(metrics_path),
    ]) == 0
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        np.testing.assert_allclose(trajectory['time'], [0, 0.02, 0.04, 0.06])
        assert trajectory['joint_position'].shape == (4, 29)
        assert trajectory['reference'].shape == (4, 71)
        assert not np.array_equal(trajectory['joint_position'][-1], trajectory['reference'][-1, 7:36])
    with pytest.raises(ValueError, match='must not exist'):
        sim2sim.validate_output(trajectory_path)


@pytest.mark.parametrize('goal_width,output_name', [(74, 'action'), (146, 'wrong_action')])
def test_incompatible_onnx_contract_rejected(tmp_path, goal_width, output_name):
    import onnx
    from onnx import TensorProto, helper

    policy_path = tmp_path / 'wrong.onnx'
    graph = helper.make_graph(
        [helper.make_node('Constant', [], [output_name], value=helper.make_tensor(
            'constant_action', TensorProto.FLOAT, [1, 29], [0.0] * 29,
        ))], 'invalid_contract',
        [helper.make_tensor_value_info('obs_hist', TensorProto.FLOAT, [1, 3, 99]),
         helper.make_tensor_value_info('goal', TensorProto.FLOAT, [1, goal_width])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 29])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, policy_path)
    with pytest.raises(ValueError):
        sim2sim.OnnxPolicy(policy_path)
