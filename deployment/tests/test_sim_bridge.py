"""Safety helpers and temporary MuJoCo models; never initialize DDS here."""

import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deployment.sim_bridge import (
    BUNDLE_FILES,
    G1_JOINT_NAMES,
    JOINT_COUNT,
    PHYSICS_DT,
    SimulationBridge,
    SimulationBundle,
    body_angular_velocity,
    build_model,
    compute_torque,
    decode_command,
    map_joints,
    reset_from_reference,
    validate_dds_isolation,
)


def refresh_hashes(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in BUNDLE_FILES
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def bundle_directory(tmp_path: Path) -> Path:
    manifest = {
        "schema_version": 1, "control_dt": 0.02, "frame_count": 3,
        "history_length": 1, "goal_frames": 1, "goal_dim": 73,
        "observation_dim": 99, "action_dim": 29,
        "policy_file": "policy.onnx", "reference_file": "reference.bin", "goals_file": "goals.bin",
        "joint_names": list(G1_JOINT_NAMES),
        "default_joint_position": [0.0] * JOINT_COUNT,
        "kp": [40.0] * JOINT_COUNT, "kd": [2.0] * JOINT_COUNT,
        "effort_limit": [5.0] * JOINT_COUNT, "velocity_limit": [10.0] * JOINT_COUNT,
        "action_scale": [0.1] * JOINT_COUNT,
        "joint_lower": [-1.0] * JOINT_COUNT, "joint_upper": [1.0] * JOINT_COUNT,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    reference = np.zeros((3, 71), dtype="<f4")
    reference[:, 2] = 2.0
    reference[:, 3] = 1.0
    reference[:, 7:36] = np.linspace(-0.2, 0.2, JOINT_COUNT)
    reference[:, 36:65] = np.linspace(-0.1, 0.1, JOINT_COUNT)
    reference.tofile(tmp_path / "reference.bin")
    np.zeros((3, 73), dtype="<f4").tofile(tmp_path / "goals.bin")
    (tmp_path / "policy.onnx").write_bytes(b"hash-only fixture; not an executable ONNX policy")
    (tmp_path / "parity.json").write_text("{}", encoding="ascii")
    refresh_hashes(tmp_path)
    return tmp_path


@pytest.fixture
def bundle(bundle_directory: Path) -> SimulationBundle:
    return SimulationBundle.load(bundle_directory)


@pytest.fixture
def simulation_clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    clock = SimpleNamespace(seconds=100.0)

    def advance(seconds: float) -> None:
        clock.seconds += seconds

    monkeypatch.setattr("deployment.sim_bridge.time", SimpleNamespace(
        monotonic=lambda: clock.seconds, sleep=advance,
    ))
    return clock


def make_message(**motor_overrides: float) -> SimpleNamespace:
    values = {"mode": 1, "q": 0.2, "dq": 0.0, "kp": 40.0, "kd": 2.0, "tau": 0.0}
    values.update(motor_overrides)
    return SimpleNamespace(mode_pr=0, motor_cmd=[SimpleNamespace(**values) for _ in range(JOINT_COUNT)])


def write_model(directory: Path, *, explicit_self_contact: bool = False) -> Path:
    """Reverse joint order and put another joint before the root to catch offsets."""
    body_xml = "\n".join(
        f'<body name="link_{index}" pos="{0.05 * index} 0 0">'
        f'<joint name="{G1_JOINT_NAMES[index]}" axis="0 1 0"/>'
        '<geom type="sphere" size="0.02" mass="0.05"/></body>'
        for index in reversed(range(JOINT_COUNT))
    )
    contact_xml = ('<contact><pair geom1="root_geom" geom2="root_mesh"/></contact>'
                   if explicit_self_contact else "")
    # The mesh and body include must resolve from the model, not the caller CWD.
    mesh_directory = directory / "meshes"
    mesh_directory.mkdir()
    (mesh_directory / "tetra.obj").write_text(
        "v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n", encoding="ascii",
    )
    (directory / "robot.xml").write_text(
        '<mujoco><body name="root" pos="0 0 2"><freejoint name="root_free"/>'
        '<geom name="root_geom" type="box" size=".1 .1 .1" mass="1" contype="0" conaffinity="0"/>'
        '<geom name="root_mesh" type="mesh" mesh="tetra" mass=".01"/>'
        f'{body_xml}</body></mujoco>', encoding="ascii",
    )
    path = directory / "model.xml"
    path.write_text(
        '<mujoco><compiler meshdir="meshes"/>'
        '<asset><mesh name="tetra" file="tetra.obj"/></asset><worldbody>'
        '<body name="other" pos="-10 0 1"><joint name="other_joint"/>'
        '<geom type="sphere" size=".1" mass=".1"/></body>'
        '<include file="robot.xml"/></worldbody>'
        f'<actuator><position joint="{G1_JOINT_NAMES[0]}" kp="1000"/></actuator>'
        f'{contact_xml}</mujoco>', encoding="ascii",
    )
    return path


def test_bundle_loads_little_endian_rows(bundle: SimulationBundle) -> None:
    assert bundle.reference.shape == (3, 71)
    np.testing.assert_allclose(bundle.reference[0, 7:36], np.linspace(-0.2, 0.2, JOINT_COUNT))


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"control_dt": 0.005}, {"frame_count": 4},
    {"joint_names": ["same"] * JOINT_COUNT}, {"kp": [-1.0] * JOINT_COUNT},
    {"kd": [math.nan] * JOINT_COUNT}, {"joint_lower": [2.0] * JOINT_COUNT},
    {"kp": [0.0] * JOINT_COUNT}, {"history_length": 0}, {"goal_frames": True},
    {"goal_dim": 74}, {"observation_dim": 98}, {"action_dim": 28},
    {"policy_file": "alternate.onnx"}, {"reference_file": "../reference.bin"},
    {"goals_file": "alternate.bin"}, {"joint_names": list(reversed(G1_JOINT_NAMES))},
    {"sha256": {}}, {"sha256": {"../reference.bin": "0" * 64}},
])
def test_bundle_rejects_contract_mismatch(bundle_directory: Path, change: dict) -> None:
    manifest_path = bundle_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(change)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        SimulationBundle.load(bundle_directory)


def test_bundle_rejects_bad_reference_quaternion(bundle_directory: Path) -> None:
    reference = np.zeros((3, 71), dtype="<f4")
    reference.tofile(bundle_directory / "reference.bin")
    refresh_hashes(bundle_directory)
    with pytest.raises(ValueError, match="quaternions"):
        SimulationBundle.load(bundle_directory)


@pytest.mark.parametrize("filename", BUNDLE_FILES)
def test_bundle_rejects_tampered_payload(bundle_directory: Path, filename: str) -> None:
    payload_path = bundle_directory / filename
    payload = bytearray(payload_path.read_bytes())
    payload[0] ^= 1
    payload_path.write_bytes(payload)
    with pytest.raises(ValueError, match=f"SHA256 mismatch for {filename}"):
        SimulationBundle.load(bundle_directory)


@pytest.mark.parametrize("filename", BUNDLE_FILES)
def test_bundle_rejects_symlink_payload(bundle_directory: Path, filename: str) -> None:
    payload_path = bundle_directory / filename
    target_path = bundle_directory / f"actual_{filename}"
    payload_path.rename(target_path)
    payload_path.symlink_to(target_path)
    with pytest.raises(ValueError, match="regular file"):
        SimulationBundle.load(bundle_directory)


@pytest.mark.parametrize("invalid_values", [np.zeros((3, 72), dtype="<f4"),
                                           np.full((3, 73), np.nan, dtype="<f4")])
def test_bundle_rejects_hashed_invalid_goals(bundle_directory: Path, invalid_values: np.ndarray) -> None:
    invalid_values.tofile(bundle_directory / "goals.bin")
    refresh_hashes(bundle_directory)
    with pytest.raises(ValueError, match="goals.bin"):
        SimulationBundle.load(bundle_directory)


@pytest.mark.parametrize("interface,domain", [("eth0", 42), ("lo", 0), ("lo", -1), ("lo", 233), ("lo", True)])
def test_dds_gating_rejects_robot_or_default_domains(monkeypatch: pytest.MonkeyPatch,
                                                    interface: str, domain: int) -> None:
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    with pytest.raises(ValueError):
        validate_dds_isolation(interface, domain)


def test_dds_gating_rejects_all_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    validate_dds_isolation("lo", 42)
    monkeypatch.setenv("CYCLONEDDS_URI", "<CycloneDDS><Domain Id='any'/></CycloneDDS>")
    with pytest.raises(ValueError, match="Unset CYCLONEDDS_URI"):
        validate_dds_isolation("lo", 42)


def test_command_caps_all_targets_and_final_effort(bundle: SimulationBundle) -> None:
    command = decode_command(make_message(q=9, dq=99, kp=900, kd=90, tau=50), bundle, 1.0)
    np.testing.assert_array_equal(command.position, bundle.joint_upper)
    np.testing.assert_array_equal(command.velocity, bundle.velocity_limit)
    np.testing.assert_array_equal(command.kp, bundle.kp)
    np.testing.assert_array_equal(command.kd, bundle.kd)
    np.testing.assert_array_equal(command.torque, bundle.effort_limit)
    applied, stale, requested = compute_torque(bundle, command, np.zeros(29), np.zeros(29), 1.01, 0.1)
    assert not stale
    assert np.all(requested > bundle.effort_limit)
    np.testing.assert_array_equal(applied, bundle.effort_limit)


@pytest.mark.parametrize("field,value", [
    ("q", math.nan), ("dq", math.inf), ("tau", -math.inf), ("kp", -1),
    ("kd", -1), ("kp", math.nan), ("kd", math.inf), ("mode", 2),
])
def test_command_rejects_nonfinite_or_unsafe_values(bundle: SimulationBundle, field: str, value: float) -> None:
    with pytest.raises(ValueError):
        decode_command(make_message(**{field: value}), bundle, 1.0)


def test_command_rejects_ab_mode(bundle: SimulationBundle) -> None:
    message = make_message()
    message.mode_pr = 1
    with pytest.raises(ValueError, match="mode_pr"):
        decode_command(message, bundle, 1.0)


def test_disabled_motor_has_no_pd_or_feedforward(bundle: SimulationBundle) -> None:
    command = decode_command(make_message(mode=0, tau=5), bundle, 1.0)
    applied, stale, _ = compute_torque(bundle, command, np.ones(29), np.ones(29), 1.01, 0.1)
    assert not stale
    np.testing.assert_array_equal(applied, np.zeros(29))


@pytest.mark.parametrize("receipt_time", [None, 1.0, 2.0])
def test_watchdog_damps_without_reusing_old_position(bundle: SimulationBundle, receipt_time: float | None) -> None:
    command = None if receipt_time is None else decode_command(make_message(q=1, tau=5), bundle, receipt_time)
    velocity = np.linspace(-5, 5, JOINT_COUNT)
    applied, stale, _ = compute_torque(bundle, command, np.zeros(29), velocity, 1.5, 0.1)
    assert stale
    np.testing.assert_allclose(applied, np.clip(-bundle.kd * velocity, -bundle.effort_limit, bundle.effort_limit))
    assert np.all(applied * velocity <= 0)


@pytest.fixture
def simulation(bundle: SimulationBundle, tmp_path: Path):
    pytest.importorskip("mujoco")
    path = write_model(tmp_path)
    model, data, mapping = build_model(path, bundle)
    return bundle, model, data, mapping


def test_model_compiles_repeatedly_without_specification_corruption(bundle: SimulationBundle,
                                                                   tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    path = write_model(tmp_path)
    for iteration in range(100):
        model, data, mapping = build_model(path, bundle)
        assert model.nu == 0
        assert model.geom_contype[model.geom("supermimic_bridge_ground").id] == 2
        assert model.geom_contype[model.geom("root_geom").id] == 1
        np.testing.assert_allclose(data.qpos[mapping.qpos_addresses], bundle.reference[0, 7:36])
        del model, data, mapping
        if iteration % 10 == 0:
            gc.collect()


def test_named_mapping_and_reference_reset_do_not_assume_offsets(simulation) -> None:
    bundle, model, data, mapping = simulation
    assert mapping.root_qpos_address == 1
    assert mapping.root_dof_address == 1
    assert np.all(np.diff(mapping.qpos_addresses) == -1)
    np.testing.assert_allclose(data.qpos[mapping.qpos_addresses], bundle.reference[0, 7:36])
    np.testing.assert_allclose(data.qvel[mapping.dof_addresses], bundle.reference[0, 36:65])
    assert data.time == 0
    assert model.nu == 0  # Native position servos were removed, not set to ctrl=0.
    assert model.opt.timestep == PHYSICS_DT
    np.testing.assert_allclose(model.dof_armature[mapping.dof_addresses], bundle.kp / (20 * math.pi) ** 2)
    with pytest.raises(ValueError, match="Missing manifest joint"):
        map_joints(model, tuple(["not_a_joint"] + list(bundle.joint_names[1:])))


def test_robot_self_collision_masks_and_floor(simulation) -> None:
    import mujoco

    _, model, data, mapping = simulation
    floor = model.geom("supermimic_bridge_ground").id
    assert model.geom_contype[floor] == 2
    assert model.geom_conaffinity[floor] == 1
    root_geoms = np.flatnonzero(model.geom_bodyid == mapping.root_body_id)
    assert np.all(model.geom_contype[root_geoms] == 1)
    assert np.all(model.geom_conaffinity[root_geoms] == 2)
    assert model.body_contype[mapping.root_body_id] == 1
    assert model.body_conaffinity[mapping.root_body_id] == 2
    data.qpos[mapping.root_qpos_address + 2] = 0.05
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert all(floor in (contact.geom1, contact.geom2) for contact in data.contact)


def test_connect_rejects_nonloopback_before_importing_sdk(simulation) -> None:
    bundle, model, data, mapping = simulation
    bridge = SimulationBridge(bundle, model, data, mapping)
    with pytest.raises(ValueError, match="interface 'lo'"):
        bridge.connect("eth0", 42)
    assert bridge.publisher is None
    assert bridge.subscriber is None


def test_explicit_self_contact_fails_closed(bundle: SimulationBundle, tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    path = write_model(tmp_path, explicit_self_contact=True)
    with pytest.raises(ValueError, match="contact pairs"):
        build_model(path, bundle)


def test_reset_world_angular_velocity_and_body_gyro(simulation) -> None:
    import mujoco

    bundle, model, data, mapping = simulation
    frame = bundle.reference[0].astype(np.float64).copy()
    frame[3:7] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    frame[65:68] = [1.0, 2.0, 3.0]
    frame[68:71] = [4.0, 5.0, 6.0]
    reset_from_reference(model, data, mapping, frame)
    velocity_world = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, mapping.root_body_id, velocity_world, 0)
    np.testing.assert_allclose(velocity_world[:3], frame[68:71], atol=1e-10)
    # The angular component rotates into body coordinates; it is NOT linear qvel.
    np.testing.assert_allclose(body_angular_velocity(model, data, mapping.root_body_id), [5, -4, 6], atol=1e-10)
    np.testing.assert_allclose(data.qvel[mapping.root_dof_address:mapping.root_dof_address + 3], frame[65:68])


def test_invalid_message_disarms_mailbox_and_fault_injection(simulation) -> None:
    bundle, model, data, mapping = simulation
    bridge = SimulationBridge(bundle, model, data, mapping)
    bridge.receive_command(make_message())
    assert bridge.command is not None
    bridge.receive_command(make_message(kp=math.nan))
    assert bridge.command is None
    assert bridge.rejected_commands == 1
    bridge.drop_command_after = 0
    bridge.receive_command(make_message())
    assert bridge.command is None
    assert bridge.dropped_commands == 1


def test_freeze_publishes_increasing_ticks_without_advancing_phase(simulation, simulation_clock) -> None:
    bundle, model, data, mapping = simulation
    bridge = SimulationBridge(bundle, model, data, mapping, freeze_before_command=True)
    ticks = []
    bridge.publisher = SimpleNamespace()
    bridge.publish_state = lambda: ticks.append(bridge.tick)
    initial_qpos = data.qpos.copy()
    metrics = bridge.run(0.015)
    assert metrics["frozen_steps"] > 0
    assert metrics["physics_steps"] == 0
    assert data.time == 0
    assert len(ticks) > 1
    assert np.all(np.diff(ticks) == 2)
    np.testing.assert_array_equal(data.qpos, initial_qpos)
    assert metrics["joint_position_rmse"] is None
    assert metrics["success_evaluated"] is False


def test_offline_loop_advances_physics_and_records_real_errors(simulation, simulation_clock) -> None:
    bundle, model, data, mapping = simulation
    bridge = SimulationBridge(bundle, model, data, mapping, command_timeout=0.005)
    bridge.receive_command(make_message(q=0.5))
    metrics = bridge.run(0.08)
    assert metrics["physics_steps"] > 0
    assert metrics["stale_steps"] > 0
    assert metrics["clipped_steps"] > 0
    assert metrics["simulation_seconds"] > 0
    assert len(metrics["joint_position_rmse"]) == 29
    assert metrics["root_position_rmse"][2] > 0
    assert metrics["maximum_applied_torque"] <= 5.0
    assert metrics["reference_exhausted"] is True
    assert metrics["final_reference_frame"] == 2
    assert metrics["success_evaluated"] is False


def test_drop_state_fault_stops_publication_without_advancing_frozen_phase(simulation, simulation_clock) -> None:
    bundle, model, data, mapping = simulation
    bridge = SimulationBridge(bundle, model, data, mapping, freeze_before_command=True,
                              drop_state_after=0.005)
    ticks = []
    bridge.publisher = SimpleNamespace()
    bridge.publish_state = lambda: ticks.append(bridge.tick)
    metrics = bridge.run(0.02)
    assert ticks == [0, 2, 4]
    assert metrics["frozen_steps"] > len(ticks)
    assert data.time == 0


def test_lowstate_fields_and_body_gyro_without_sdk(simulation) -> None:
    bundle, model, data, mapping = simulation
    motors = [SimpleNamespace(mode=0, q=99, dq=99, ddq=99, tau_est=99, temperature=[99, 99]) for _ in range(35)]
    state = SimpleNamespace(motor_state=motors, imu_state=SimpleNamespace(
        quaternion=[0] * 4, gyroscope=[0] * 3, accelerometer=[0] * 3,
    ))
    bridge = SimulationBridge(bundle, model, data, mapping)
    publications = []
    bridge.publisher = SimpleNamespace(Write=lambda message: publications.append(message))
    bridge.low_state = state
    bridge.crc = SimpleNamespace(Crc=lambda message: 123)
    bridge.tick = 12
    bridge.publish_state()
    assert publications == [state]
    assert state.mode_pr == 0
    assert state.mode_machine == 5
    assert state.tick == 12
    assert state.crc == 123
    np.testing.assert_allclose([motor.q for motor in motors[:29]], bundle.reference[0, 7:36])
    assert all(motor.temperature == [0, 0] for motor in motors)
    assert all(motor.mode == 0 and motor.q == 0 for motor in motors[29:])
    np.testing.assert_allclose(state.imu_state.quaternion, [1, 0, 0, 0])
