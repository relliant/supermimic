"""Focused, CPU-only deployment conformance tests; no Isaac/FK/simulation."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from deployment.common import contract
from deployment.export.policy import (
    Policy, RunningNorm, build_parity_samples, export_bundle, export_onnx,
    load_policy, validate_onnx_parity,
)
from supermimic.rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp

REPOSITORY = Path(__file__).resolve().parents[2]


def create_synthetic_bundle_fixture(directory: str | Path) -> Path:
    """Create a reproducible H=10/K=10 *untrained* native-conformance bundle.

    The directory must be fresh. Inputs remain beside ``bundle/`` so native
    integration checks can inspect/re-export them without a separate script.
    The tiny MJCF has G1 joint *names*, not G1 geometry or authentic ranges;
    neither it nor the random checkpoint is suitable for closed-loop testing.
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    checkpoint_path = directory / "SYNTHETIC_UNTRAINED.pt"
    motion_path = directory / "SYNTHETIC_motion.csv"
    mjcf_path = directory / "SYNTHETIC_named_joints.xml"
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(20261026)
        policy = Policy(history_length=10, goal_frames=10).eval()
        policy.dec[-1].weight.mul_(0.01)
        policy.dec[-1].bias.zero_()
        for normalizer in (policy.obs_norm, policy.goal_norm):
            synthetic_features = 0.8 * torch.randn(64, normalizer.mean.numel()) + 0.15
            normalizer.mean.copy_(synthetic_features.mean(0))
            normalizer.var.copy_(synthetic_features.var(0, unbiased=False))
            normalizer.count.fill_(64)
        torch.save({
            "pi": policy.state_dict(),
            "args": {"hist": 10, "goal_frames": 10, "amp": "none", "synthetic_untrained": True},
        }, checkpoint_path)

    time = torch.arange(31, dtype=torch.float32) / 30
    position = torch.stack((0.2 * time, 0.05 * torch.sin(time * 2), 0.8 + 0.03 * torch.cos(time)), -1)
    rotation_vectors = torch.stack((0.2 + 0.1 * time, -0.15 + 0.2 * time, 0.7 + 0.4 * time), -1)
    quaternion = mat_to_quat(so3_exp(rotation_vectors))
    joint_phase = torch.linspace(0, 2, 29)
    joints = torch.tensor(contract.DEFAULT_JOINT_POSITION)[None] + 0.1 * torch.sin(3 * time[:, None] + joint_phase)
    raw_motion = torch.cat((position, quaternion[:, 1:], quaternion[:, :1], joints), -1).numpy()
    np.savetxt(motion_path, raw_motion, delimiter=",")
    bodies = [
        f'<body name="synthetic_body_{index}" pos="{0.1 * index} 0 1">'
        f'<joint name="{name}" type="hinge" limited="true" range="-3 3"/>'
        '<geom type="sphere" size="0.01" mass="0.1"/></body>'
        for index, name in enumerate(contract.G1_JOINT_NAMES)
    ]
    mjcf_path.write_text(
        '<mujoco model="SYNTHETIC_NOT_G1_GEOMETRY"><compiler angle="radian"/>'
        '<worldbody>' + "".join(bodies) + '</worldbody></mujoco>', encoding="utf-8",
    )
    bundle_path = directory / "bundle"
    export_bundle(checkpoint_path, motion_path, bundle_path, mjcf_path, parity_samples=12)
    return bundle_path


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous_count = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_count))
    yield
    torch.set_num_threads(previous_count)


@pytest.fixture(scope="module")
def original_modules():
    """Load the real training source under an isolated package with FK stubbed.

    Importing networks normally imports mdp -> kinematics -> robot -> Isaac.
    The stub supplies only the unused type and inert reference-body outputs;
    all policy, observation, motion resampling and goal code is unmodified.
    """
    package_name = "_deployment_original_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPOSITORY / "supermimic")]
    sys.modules[package_name] = package
    kinematics = types.ModuleType(f"{package_name}.kinematics")

    class InertKinematics:
        def forward(self, position, rotation, linear_velocity, angular_velocity, joints, joint_velocity):
            frame_count = position.shape[0]
            return (
                torch.zeros(frame_count, 14, 3), torch.zeros(frame_count, 14, 3, 3),
                torch.zeros(frame_count, 14, 3), torch.zeros(frame_count, 14, 3),
            )

    kinematics.G1Kinematics = InertKinematics
    sys.modules[kinematics.__name__] = kinematics
    modules = {"kinematics": kinematics}
    try:
        for name in ("rotations", "motion", "mdp", "networks"):
            specification = importlib.util.spec_from_file_location(
                f"{package_name}.{name}", REPOSITORY / "supermimic" / f"{name}.py"
            )
            module = importlib.util.module_from_spec(specification)
            sys.modules[specification.name] = module
            specification.loader.exec_module(module)
            modules[name] = module
        yield types.SimpleNamespace(**modules)
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                del sys.modules[name]


@pytest.fixture(scope="module")
def training_policy(original_modules):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(371)
        policy = original_modules.networks.Policy(99, 146, hist=3).eval()
        policy.obs_norm.update(torch.randn(32, 99) * 0.7 + 0.2)
        policy.goal_norm.update(torch.randn(32, 146) * 1.2 - 0.3)
    return policy


@pytest.fixture
def motion_csv(tmp_path):
    source_times = torch.arange(7, dtype=torch.float32) / 30
    position = torch.stack((0.3 * source_times, -0.2 * source_times, 0.8 + 0.1 * source_times.square()), -1)
    rotation_vectors = torch.stack((0.2 + source_times, -0.15 + 0.3 * source_times, 0.7 + 1.3 * source_times), -1)
    quaternion = mat_to_quat(so3_exp(rotation_vectors))
    quaternion[3] *= -1  # Exercise SLERP's shortest-arc sign correction.
    joints = source_times[:, None] * torch.linspace(-0.8, 1.2, 29)[None] + torch.linspace(-0.2, 0.3, 29)[None]
    raw = torch.cat((position, quaternion[:, 1:], quaternion[:, :1], joints), -1).numpy()
    path = tmp_path / "motion.csv"
    np.savetxt(path, raw, delimiter=",")
    return path


@pytest.fixture
def checkpoint(tmp_path, training_policy):
    path = tmp_path / "checkpoint.pt"
    # The deliberately irrelevant world-model key proves pi is the only source.
    torch.save({"pi": training_policy.state_dict(), "wm": {"garbage": torch.ones(1)},
                "args": {"hist": 3, "goal_frames": 2, "amp": "bf16"}}, path)
    return path


@pytest.fixture
def mjcf(tmp_path):
    bodies = []
    for index, name in enumerate(contract.G1_JOINT_NAMES):
        bodies.append(
            f'<body name="body_{index}" pos="{index * 0.1} 0 1">'
            f'<joint name="{name}" range="-90 90"/>'
            '<geom type="sphere" size="0.01" mass="0.1"/></body>'
        )
    path = tmp_path / "robot.xml"
    path.write_text('<mujoco><compiler angle="degree"/><worldbody>' + "".join(bodies) + '</worldbody></mujoco>')
    return path


def test_mat6d_identity_and_nontrivial_interleaved_columns():
    torch.testing.assert_close(mat_to_6d(torch.eye(3)), torch.tensor([1., 0., 0., 1., 0., 0.]))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(mat_to_6d(rotation), torch.tensor([0., -1., 1., 0., 0., 0.]))
    arbitrary_rotation = so3_exp(torch.tensor([0.4, -0.2, 0.7]))
    expected = torch.stack((
        arbitrary_rotation[0, 0], arbitrary_rotation[0, 1], arbitrary_rotation[1, 0],
        arbitrary_rotation[1, 1], arbitrary_rotation[2, 0], arbitrary_rotation[2, 1],
    ))
    torch.testing.assert_close(mat_to_6d(arbitrary_rotation), expected)


def test_observation_matches_original_mdp_without_extra_scale(original_modules):
    root_rotation = so3_exp(torch.tensor([[0.3, -0.5, 0.8], [-0.6, 0.2, -0.4]]))
    root_quaternion = mat_to_quat(root_rotation)
    root_rotation = quat_to_mat(root_quaternion)
    reference_quaternion = mat_to_quat(so3_exp(torch.tensor([[0.1, 0.2, -0.4], [0.5, 0.1, 0.2]])))
    position = torch.linspace(-0.5, 0.6, 58).reshape(2, 29)
    velocity = position * 4.2
    angular_velocity_body = torch.tensor([[1.2, -0.7, 0.2], [-0.4, 0.8, 1.5]])
    previous_action = position * -0.8
    default_position = torch.tensor(contract.DEFAULT_JOINT_POSITION)
    observation = contract.build_observation(
        position, velocity, root_quaternion, angular_velocity_body,
        previous_action, reference_quaternion, default_position,
    )
    mdp = original_modules.mdp.MDP.__new__(original_modules.mdp.MDP)
    mdp.default_q = default_position
    mdp.motion = types.SimpleNamespace(root_R=quat_to_mat(reference_quaternion))
    state = (
        torch.zeros(2, 3), root_rotation, torch.zeros(2, 3),
        (root_rotation @ angular_velocity_body.unsqueeze(-1)).squeeze(-1), position, velocity,
    )
    expected = mdp.obs(state, previous_action, torch.zeros(2, 33), torch.arange(2))
    torch.testing.assert_close(observation, expected, rtol=1e-6, atol=3e-7)
    torch.testing.assert_close(observation[:, 29:58], velocity, rtol=0, atol=0)
    numpy_observation = contract.build_observation(
        position.numpy(), velocity.numpy(), root_quaternion.numpy(), angular_velocity_body.numpy(),
        previous_action.numpy(), reference_quaternion.numpy(), default_position.numpy(),
    )
    np.testing.assert_array_equal(numpy_observation, observation.numpy())


def test_motion_resampling_gradients_and_goals_match_original(motion_csv, original_modules):
    reference = contract.load_reference(motion_csv)
    original = original_modules.motion.MotionLib(
        [str(motion_csv)], original_modules.kinematics.G1Kinematics(), "cpu",
    )
    assert reference.dtype == np.float32
    assert reference.shape == (11, 71)
    for actual, expected in (
        (reference[:, :3], original.root_p), (reference[:, 7:36], original.q),
        (reference[:, 36:65], original.qd), (reference[:, 65:68], original.root_v),
        (reference[:, 68:71], original.root_w),
    ):
        np.testing.assert_array_equal(actual, expected.numpy())
    torch.testing.assert_close(quat_to_mat(torch.from_numpy(reference[:, 3:7])), original.root_R, rtol=0, atol=0)
    np.testing.assert_allclose(reference[:, 65], 0.3, atol=2e-7)
    np.testing.assert_allclose(reference[:, 36:65], np.linspace(-0.8, 1.2, 29)[None].repeat(11, axis=0), atol=3e-6)
    for goal_frames in (1, 4, 15):
        expected = original.goal(torch.arange(reference.shape[0]), goal_frames)
        actual = contract.build_goals(reference, goal_frames)
        np.testing.assert_array_equal(actual, expected.numpy())
        torch.testing.assert_close(contract.build_goals(torch.from_numpy(reference), goal_frames), expected, rtol=0, atol=0)
    goals = contract.build_goals(reference, 4).reshape(11, 4, 73)
    np.testing.assert_array_equal(goals[-1, :, :3], np.zeros((4, 3)))
    np.testing.assert_array_equal(goals[-1, 0], goals[-1, 3])
    np.testing.assert_array_equal(goals[0, 0, 15:44], reference[1, 7:36])
    np.testing.assert_array_equal(goals[0, 0, 44:73], reference[1, 36:65] * np.float32(0.1))


def test_future_endpoint_and_history_semantics():
    np.testing.assert_array_equal(contract.future_frames(np.array([0, 2, 3]), 4, 3), [[1, 2, 3], [3, 3, 3], [3, 3, 3]])
    initial = np.arange(99, dtype=np.float32)
    history = contract.initialize_history(initial, 3)
    np.testing.assert_array_equal(history, np.tile(initial, (3, 1)))
    for offset in (100, 200, 300):
        previous = history.copy()
        history = contract.update_history(history, initial + offset)
        np.testing.assert_array_equal(history[:-1], previous[1:])
        np.testing.assert_array_equal(history[-1], initial + offset)
    np.testing.assert_array_equal(history[:, 0], [100, 200, 300])
    one_frame = contract.update_history(contract.initialize_history(initial, 1), initial + 5)
    np.testing.assert_array_equal(one_frame, (initial + 5)[None])
    with pytest.raises(ValueError, match="outside"):
        contract.future_frames(np.array([4]), 4, 2)
    with pytest.raises(ValueError, match="integers"):
        contract.future_frames(np.array([0.5]), 4, 2)
    with pytest.raises(ValueError, match="matching"):
        contract.update_history(np.zeros((2, 3, 99)), np.zeros(99))


def test_frozen_policy_exact_keys_and_float32_parity(checkpoint, training_policy):
    policy, metadata = load_policy(checkpoint)
    assert list(policy.state_dict()) == list(training_policy.state_dict())
    assert metadata["training_amp"] == "bf16"
    assert policy.obs_norm.eps == 0.01
    generator = torch.Generator().manual_seed(919)
    with torch.inference_mode():
        for _ in range(3):
            history = torch.randn(4, 3, 99, generator=generator)
            goal = torch.randn(4, 146, generator=generator)
            torch.testing.assert_close(policy(history, goal), training_policy(history, goal), rtol=0, atol=0)
    norm = RunningNorm(2)
    norm.var.zero_()
    torch.testing.assert_close(norm(torch.tensor([0.02, -0.03])), torch.tensor([2., -3.]))


@pytest.mark.parametrize("corruption, message", [
    ("missing_pi", "'pi'"), ("zero_count", "positive"), ("negative_var", "nonnegative"),
    ("nonfinite", "nonfinite"), ("wrong_shape", "shape"), ("extra_key", "Unexpected key"),
    ("wrong_hist", "positive integer"), ("missing_goal_frames", "positive integer"),
])
def test_checkpoint_validation_rejects_bad_contract(tmp_path, training_policy, corruption, message):
    state = {name: tensor.clone() for name, tensor in training_policy.state_dict().items()}
    contents = {"pi": state, "args": {"hist": 3, "goal_frames": 2}}
    if corruption == "missing_pi":
        contents["policy"] = contents.pop("pi")
    elif corruption == "zero_count":
        state["obs_norm.count"].zero_()
    elif corruption == "negative_var":
        state["goal_norm.var"][0] = -1
    elif corruption == "nonfinite":
        state["obs_norm.mean"][0] = float("nan")
    elif corruption == "wrong_shape":
        state["obs_norm.mean"] = torch.zeros(98)
    elif corruption == "extra_key":
        state["unexpected"] = torch.zeros(1)
    elif corruption == "wrong_hist":
        contents["args"]["hist"] = 3.5
    elif corruption == "missing_goal_frames":
        del contents["args"]["goal_frames"]
    path = tmp_path / "bad.pt"
    torch.save(contents, path)
    with pytest.raises((ValueError, RuntimeError), match=message):
        load_policy(path)


def test_checkpoint_rejects_nonprimitive_pickle_without_unsafe_fallback(tmp_path):
    checkpoint = tmp_path / "nonprimitive.pt"
    torch.save({"pi": {}, "args": {"hist": 3, "goal_frames": 2}, "unexpected_object": Path("unused")}, checkpoint)
    with pytest.raises(ValueError, match="weights_only=True"):
        load_policy(checkpoint)


def test_invalid_features_and_csv_are_rejected(tmp_path):
    arguments = [np.zeros(29), np.zeros(29), np.array([1, 0, 0, 0]), np.zeros(3),
                 np.zeros(29), np.array([1, 0, 0, 0]), contract.DEFAULT_JOINT_POSITION]
    arguments[2] = np.zeros(4)
    with pytest.raises(ValueError, match="nonzero"):
        contract.build_observation(*arguments)
    path = tmp_path / "bad.csv"
    np.savetxt(path, np.zeros((1, 36)), delimiter=",")
    with pytest.raises(ValueError, match="two rows"):
        contract.load_reference(path)
    np.savetxt(path, np.zeros((2, 36)), delimiter=",")
    with pytest.raises(ValueError, match="nonzero"):
        contract.load_reference(path)
    with pytest.raises(ValueError, match="finite"):
        contract.build_goals(np.full((2, 71), np.nan), 2)


def test_robot_constants_and_compiled_limits_match_source(mjcf):
    pytest.importorskip("mujoco")
    # Execute original constants/functions, excluding only Isaac imports and
    # the final articulation construction. No fake replacement constants.
    robot_path = REPOSITORY / "supermimic/robot.py"
    syntax = ast.parse(robot_path.read_text())
    syntax.body = [node for node in syntax.body if not (
        isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name.startswith("isaaclab") for alias in node.names
        ) or isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab")
        or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "G1_CFG" for target in node.targets)
    )]
    original = {"__file__": str(robot_path)}
    exec(compile(syntax, str(robot_path), "exec"), original)
    parameters = contract.robot_parameters(mjcf)
    assert parameters["joint_names"] == original["G1_JOINT_NAMES"]
    assert parameters["default_joint_position"] == original["default_joint_pos_list"]()
    for index, name in enumerate(parameters["joint_names"]):
        effort, velocity, armature = original["motor_of"](name)
        assert parameters["kp"][index] == original["_kp"](armature)
        assert parameters["kd"][index] == original["_kd"](armature)
        assert parameters["action_scale"][index] == original["action_scale"](name)
        assert parameters["effort_limit"][index] == effort
        assert parameters["velocity_limit"][index] == velocity
    np.testing.assert_allclose(parameters["joint_lower"], -np.pi / 2)
    np.testing.assert_allclose(parameters["joint_upper"], np.pi / 2)


def test_onnx_export_and_nontrivial_parity(tmp_path, checkpoint, motion_csv):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    policy, _ = load_policy(checkpoint)
    reference = contract.load_reference(motion_csv)
    goals = contract.build_goals(reference, 2)
    parity = build_parity_samples(policy, reference, goals, contract.DEFAULT_JOINT_POSITION, 5)
    assert parity == build_parity_samples(policy, reference, goals, contract.DEFAULT_JOINT_POSITION, 5)
    assert parity["samples"][0]["frame"] == 0
    assert parity["samples"][-1]["frame"] == len(reference) - 1
    for sample in parity["samples"]:
        observation = contract.build_observation(
            sample["joint_position"], sample["joint_velocity"], sample["root_quaternion_wxyz"],
            sample["angular_velocity_body"], sample["previous_action"],
            reference[sample["frame"], 3:7], contract.DEFAULT_JOINT_POSITION,
        )
        np.testing.assert_array_equal(observation, sample["observation"])
        np.testing.assert_array_equal(np.asarray(sample["history"]).reshape(3, 99)[-1], sample["observation"])
        assert np.linalg.norm(np.asarray(sample["root_quaternion_wxyz"])[1:]) > 0.1
    path = tmp_path / "policy.onnx"
    export_onnx(policy, path)
    metadata = validate_onnx_parity(path, parity, 3)
    assert metadata["onnxruntime_checked"]
    assert metadata["maximum_absolute_action_error"] < 1e-5
    graph = onnx.load(path)
    initializer_names = {initializer.name for initializer in graph.graph.initializer}
    assert {"obs_norm.mean", "obs_norm.var", "goal_norm.mean", "goal_norm.var"} <= initializer_names


def test_h10_k10_synthetic_native_fixture(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("mujoco")
    bundle_path = create_synthetic_bundle_fixture(tmp_path / "synthetic-untrained")
    manifest = contract.verify_bundle(bundle_path)
    assert manifest["history_length"] == 10
    assert manifest["goal_frames"] == 10 and manifest["goal_dim"] == 730
    assert manifest["frame_count"] == 51
    checkpoint = torch.load(bundle_path.parent / "SYNTHETIC_UNTRAINED.pt", weights_only=True)
    assert checkpoint["args"]["synthetic_untrained"] is True
    parity = json.loads((bundle_path / "parity.json").read_text())
    assert len(parity["samples"]) == 12
    assert parity["samples"][0]["frame"] == 0
    assert parity["samples"][-1]["frame"] == 50
    for sample in parity["samples"]:
        assert len(sample["history"]) == 990 and len(sample["goal"]) == 730
    final_history = np.asarray(parity["samples"][-1]["history"]).reshape(10, 99)
    np.testing.assert_array_equal(final_history, [sample["observation"] for sample in parity["samples"][-10:]])
    with pytest.raises(FileExistsError):
        create_synthetic_bundle_fixture(bundle_path.parent)


def test_atomic_bundle_schema_hashes_no_overwrite_and_failure_cleanup(tmp_path, checkpoint, motion_csv, mjcf, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("mujoco")
    destination = tmp_path / "bundle"
    manifest = export_bundle(checkpoint, motion_csv, destination, mjcf, parity_samples=3)
    assert manifest == contract.verify_bundle(destination)
    assert set(path.name for path in destination.iterdir()) == {*contract.BUNDLE_FILES, "manifest.json"}
    assert manifest["history_length"] == 3 and manifest["goal_dim"] == 146
    assert manifest["metadata"]["checkpoint"]["sha256"] == contract.sha256_file(checkpoint)
    assert manifest["metadata"]["source_sha256"]["supermimic/networks.py"]
    for source in ("deployment/common/contract.py", "deployment/export/policy.py"):
        assert manifest["metadata"]["source_sha256"][source] == contract.sha256_file(REPOSITORY / source)
    # The relocated profile generator must still find deployment/config, not
    # silently resolve a similarly named directory relative to its new package.
    from deployment.safety.profile import main as generate_profile

    profile_path = tmp_path / "unvalidated_profile.json"
    monkeypatch.setattr(sys, "argv", ["make_safety_profile", "--bundle", str(destination),
                                     "--output", str(profile_path)])
    generate_profile()
    profile = json.loads(profile_path.read_text())
    assert profile["hardware_validated"] is False
    assert profile["safe_exit_verified"] is False
    assert profile["manifest_sha256"] == contract.sha256_file(destination / "manifest.json")
    reference = np.fromfile(destination / "reference.bin", dtype="<f4").reshape(-1, 71)
    np.testing.assert_array_equal(reference, contract.load_reference(motion_csv))
    goals = np.fromfile(destination / "goals.bin", dtype="<f4").reshape(-1, 146)
    np.testing.assert_array_equal(goals, contract.build_goals(reference, 2))
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, motion_csv, destination, mjcf)
    empty_destination = tmp_path / "empty"
    empty_destination.mkdir()
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, motion_csv, empty_destination, mjcf)
    broken_link = tmp_path / "broken-link"
    broken_link.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, motion_csv, broken_link, mjcf)
    parity_path = destination / "parity.json"
    parity_path.write_text(parity_path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        contract.verify_bundle(destination)
    altered = json.loads((destination / "manifest.json").read_text())
    altered["policy_file"] = "../outside.onnx"
    (destination / "manifest.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="policy_file"):
        contract.verify_bundle(destination)

    def fail_export(*arguments, **keywords):
        raise RuntimeError("intentional failed export")

    monkeypatch.setattr("deployment.export.policy.export_onnx", fail_export)
    failed_destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional"):
        export_bundle(checkpoint, motion_csv, failed_destination, mjcf, parity_samples=1)
    assert not failed_destination.exists()
    assert not list(tmp_path.glob(".failed.*"))
