"""Deployment feature and binary contracts, independent of Isaac Lab and FK.

Public feature builders accept NumPy arrays or CPU/GPU torch tensors and return
the same array family, using float32 features (integer future-frame indices).
Reference rows are NOT ``mdp.flatten_state`` rows: their layout is
position(3), quaternion_wxyz(4), joints(29), joint_velocities(29), world linear
velocity(3), world angular velocity(3). Rotation 6D features flatten the first
two matrix columns in row-major order, not one whole column after the other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from supermimic.rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

SCHEMA_VERSION = 1
CONTROL_DT = 0.02
OBSERVATION_DIM = 99
ACTION_DIM = 29
REFERENCE_DIM = 71
GOAL_FRAME_DIM = 73
POLICY_FILE = "policy.onnx"
REFERENCE_FILE = "reference.bin"
GOALS_FILE = "goals.bin"
PARITY_FILE = "parity.json"
BUNDLE_FILES = (POLICY_FILE, REFERENCE_FILE, GOALS_FILE, PARITY_FILE)

# Kept independent of robot.py, whose articulation configuration imports Isaac.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DEFAULT_JOINT_POSITION = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]

_JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, 0.010177520),
    "hip_roll": (139.0, 20.0, 0.025101925),
    "hip_yaw": (88.0, 32.0, 0.010177520),
    "knee": (139.0, 20.0, 0.025101925),
    "ankle_pitch": (50.0, 37.0, 2.0 * 0.003609725),
    "ankle_roll": (50.0, 37.0, 2.0 * 0.003609725),
    "waist_yaw": (88.0, 32.0, 0.010177520),
    "waist_roll": (50.0, 37.0, 2.0 * 0.003609725),
    "waist_pitch": (50.0, 37.0, 2.0 * 0.003609725),
    "shoulder_pitch": (25.0, 37.0, 0.003609725),
    "shoulder_roll": (25.0, 37.0, 0.003609725),
    "shoulder_yaw": (25.0, 37.0, 0.003609725),
    "elbow": (25.0, 37.0, 0.003609725),
    "wrist_roll": (25.0, 37.0, 0.003609725),
    "wrist_pitch": (5.0, 22.0, 0.00425),
    "wrist_yaw": (5.0, 22.0, 0.00425),
}


def _as_tensor(value: Any, name: str, width: int, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0 or tensor.shape[-1] != width:
        raise ValueError(f"{name} must have final dimension {width}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _quaternion(value: Any, name: str, device: torch.device | None = None) -> torch.Tensor:
    quaternion = _as_tensor(value, name, 4, device)
    if (torch.linalg.vector_norm(quaternion, dim=-1) < 1e-8).any():
        raise ValueError(f"{name} must have nonzero norm")
    return quaternion


def _return_like(tensor: torch.Tensor, original: Any) -> torch.Tensor | np.ndarray:
    return tensor if isinstance(original, torch.Tensor) else tensor.detach().cpu().numpy()


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_observation(
    joint_position: Any,
    joint_velocity: Any,
    root_quaternion_wxyz: Any,
    angular_velocity_body: Any,
    previous_action: Any,
    reference_quaternion_wxyz: Any,
    default_joint_position: Any,
) -> torch.Tensor | np.ndarray:
    """Build the *unscaled*, noise-free 99D training observation (zero xi).

    ``OBS_SCALE`` is declared but never applied in the training/evaluation
    source. In particular, do not multiply joint velocity by 0.1 or angular
    velocity by 0.25 here; the policy contains the checkpoint's RunningNorm.
    """
    position = _as_tensor(joint_position, "joint_position", ACTION_DIM)
    device = position.device
    velocity = _as_tensor(joint_velocity, "joint_velocity", ACTION_DIM, device)
    quaternion = _quaternion(root_quaternion_wxyz, "root_quaternion_wxyz", device)
    angular_velocity = _as_tensor(angular_velocity_body, "angular_velocity_body", 3, device)
    action = _as_tensor(previous_action, "previous_action", ACTION_DIM, device)
    reference_quaternion = _quaternion(reference_quaternion_wxyz, "reference_quaternion_wxyz", device)
    default_position = _as_tensor(default_joint_position, "default_joint_position", ACTION_DIM, device)
    rotation = quat_to_mat(quaternion)
    relative_rotation = rotation.transpose(-1, -2) @ quat_to_mat(reference_quaternion)
    batch_shape = torch.broadcast_shapes(
        position.shape[:-1], velocity.shape[:-1], quaternion.shape[:-1], angular_velocity.shape[:-1],
        action.shape[:-1], reference_quaternion.shape[:-1], default_position.shape[:-1],
    )
    observation = torch.cat(
        (
            (position - default_position).expand(*batch_shape, ACTION_DIM),
            velocity.expand(*batch_shape, ACTION_DIM),
            angular_velocity.expand(*batch_shape, 3),
            (-rotation[..., 2, :]).expand(*batch_shape, 3),
            action.expand(*batch_shape, ACTION_DIM),
            mat_to_6d(relative_rotation).expand(*batch_shape, 6),
        ),
        dim=-1,
    )
    return _return_like(observation, joint_position)


def initialize_history(observation: Any, history_length: int) -> torch.Tensor | np.ndarray:
    """Repeat the first observation across history, oldest to newest."""
    length = positive_integer(history_length, "history_length")
    features = _as_tensor(observation, "observation", OBSERVATION_DIM)
    history = features.unsqueeze(-2).expand(*features.shape[:-1], length, OBSERVATION_DIM).clone()
    return _return_like(history, observation)


def update_history(history: Any, observation: Any) -> torch.Tensor | np.ndarray:
    """Drop the oldest row and append one observation; never modify the input."""
    previous = _as_tensor(history, "history", OBSERVATION_DIM)
    current = _as_tensor(observation, "observation", OBSERVATION_DIM, previous.device)
    if previous.ndim < 2 or previous.shape[-2] < 1 or previous.shape[:-2] != current.shape[:-1]:
        raise ValueError("history must have shape [..., H, 99] matching observation [..., 99]")
    return _return_like(torch.cat((previous[..., 1:, :], current.unsqueeze(-2)), dim=-2), history)


def _gradient(values: torch.Tensor) -> torch.Tensor:
    gradient = torch.zeros_like(values)
    gradient[1:-1] = (values[2:] - values[:-2]) / (2 * CONTROL_DT)
    gradient[0] = (values[1] - values[0]) / CONTROL_DT
    gradient[-1] = (values[-1] - values[-2]) / CONTROL_DT
    return gradient


def load_reference(csv: str | Path, src_fps: float = 30.0) -> np.ndarray:
    """Resample one LAFAN CSV exactly like MotionLib, without constructing FK.

    Input has root position, quaternion **xyzw**, and 29 SDK-order joints.
    Linear/joint gradients use one-sided endpoints and central interiors; world
    angular velocities use the original torch SO(3) logarithm differences.
    """
    if not np.isfinite(src_fps) or src_fps <= 0:
        raise ValueError("src_fps must be finite and positive")
    raw = torch.tensor(np.loadtxt(csv, delimiter=",", ndmin=2), dtype=torch.float32)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 36:
        raise ValueError("motion CSV requires at least two rows and 36 columns")
    if not torch.isfinite(raw).all():
        raise ValueError("motion CSV must contain only finite values")
    source_position = raw[:, :3]
    source_quaternion = torch.cat((raw[:, 6:7], raw[:, 3:6]), dim=-1)
    _quaternion(source_quaternion, "motion quaternion")
    source_joints = raw[:, 7:36]
    duration = (raw.shape[0] - 1) / src_fps
    frame_count = int(np.floor(duration / CONTROL_DT)) + 1
    if frame_count < 2:
        raise ValueError("motion must contain at least two resampled control frames")
    time = torch.arange(frame_count, dtype=torch.float32) * CONTROL_DT * src_fps
    indices = torch.clamp(time.floor().long(), max=raw.shape[0] - 2)
    fraction = (time - indices.float()).clamp(0, 1)
    position = source_position[indices] * (1 - fraction[:, None]) + source_position[indices + 1] * fraction[:, None]
    joints = source_joints[indices] * (1 - fraction[:, None]) + source_joints[indices + 1] * fraction[:, None]
    quaternion = quat_slerp(source_quaternion[indices], source_quaternion[indices + 1], fraction)
    rotation = quat_to_mat(quaternion)
    angular_velocity = torch.zeros_like(position)
    angular_velocity[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
    angular_velocity[0] = so3_log(rotation[1] @ rotation[0].T) / CONTROL_DT
    angular_velocity[-1] = so3_log(rotation[-1] @ rotation[-2].T) / CONTROL_DT
    reference = torch.cat(
        (position, quaternion, joints, _gradient(joints), _gradient(position), angular_velocity), dim=-1
    )
    if not torch.isfinite(reference).all():
        raise ValueError("motion preprocessing produced nonfinite values")
    return reference.numpy()


def future_frames(frames: Any, frame_count: int, goal_frames: int) -> torch.Tensor | np.ndarray:
    """Return f+1..f+K, clamped to the inclusive final frame of one clip."""
    count = positive_integer(frame_count, "frame_count")
    horizon = positive_integer(goal_frames, "goal_frames")
    source = torch.as_tensor(frames)
    if source.dtype == torch.bool or source.is_floating_point() or source.is_complex():
        raise ValueError("frames must contain integers")
    indices = source.to(dtype=torch.long)
    if (indices < 0).any() or (indices >= count).any():
        raise ValueError("frames are outside the reference clip")
    future = indices.unsqueeze(-1) + torch.arange(1, horizon + 1, device=indices.device)
    return _return_like(future.clamp(max=count - 1), frames)


def build_goals(reference: Any, goal_frames: int, *, hold_final_velocity: bool = False) -> torch.Tensor | np.ndarray:
    """Build MotionLib.goal, holding pose but zeroing velocities beyond the clip.

    ``hold_final_velocity`` reproduces older checkpoints trained before the
    stationary clip-tail fix; recorded endpoint velocities are always retained.
    """
    horizon = positive_integer(goal_frames, "goal_frames")
    rows = _as_tensor(reference, "reference", REFERENCE_DIM)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError("reference must have shape [N, 71] with N >= 1")
    _quaternion(rows[:, 3:7], "reference quaternion")
    position, rotation = rows[:, :3], quat_to_mat(rows[:, 3:7])
    inverse_rotation = rotation.transpose(-1, -2)
    linear_velocity_local = (inverse_rotation @ rows[:, 65:68, None]).squeeze(-1)
    angular_velocity_local = (inverse_rotation @ rows[:, 68:71, None]).squeeze(-1)
    frame_indices = torch.arange(rows.shape[0], device=rows.device)
    future = future_frames(frame_indices, rows.shape[0], horizon)
    requested = frame_indices[:, None] + torch.arange(1, horizon + 1, device=rows.device)
    recorded = (requested < rows.shape[0]).unsqueeze(-1)
    linear_future = linear_velocity_local[future]
    angular_future = angular_velocity_local[future]
    joint_future = rows[future, 36:65]
    if not hold_final_velocity:
        linear_future = torch.where(recorded, linear_future, 0.0)
        angular_future = torch.where(recorded, angular_future, 0.0)
        joint_future = torch.where(recorded, joint_future, 0.0)
    heading_inverse = yaw_mat(rotation).transpose(-1, -2).unsqueeze(1)
    displacement = (heading_inverse @ (position[future] - position.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    orientation = mat_to_6d(heading_inverse @ rotation[future])
    features = torch.cat(
        (
            displacement, orientation, linear_future, angular_future,
            rows[future, 7:36], joint_future * 0.1,
        ),
        dim=-1,
    ).reshape(rows.shape[0], GOAL_FRAME_DIM * horizon)
    return _return_like(features, reference)


def robot_parameters(mjcf: str | Path) -> dict[str, Any]:
    """Read compiled hard joint ranges; all controller constants match robot.py.

    MuJoCo is an export-time optional dependency. Compiling instead of parsing
    XML honors included files, inherited defaults, and compiler angle units.
    No simulation steps, renderer, or FK are executed.
    """
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("Reading MJCF joint limits requires the optional 'mujoco' package") from error
    model = mujoco.MjModel.from_xml_path(str(Path(mjcf).resolve()))
    parameters: dict[str, Any] = {
        "joint_names": list(G1_JOINT_NAMES),
        "default_joint_position": list(DEFAULT_JOINT_POSITION),
        **{name: [] for name in (
            "action_scale", "kp", "kd", "effort_limit", "velocity_limit", "joint_lower", "joint_upper"
        )},
    }
    natural_frequency = 10 * 2.0 * 3.1415926535
    for joint_name in G1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MJCF is missing required joint {joint_name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[joint_id]:
            raise ValueError(f"MJCF joint {joint_name!r} must be a limited hinge")
        lower, upper = model.jnt_range[joint_id].tolist()
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"MJCF joint {joint_name!r} has invalid limits")
        motor_name = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
        effort, velocity, armature = _JOINT_MOTORS[motor_name]
        stiffness = armature * natural_frequency**2
        parameters["joint_lower"].append(lower)
        parameters["joint_upper"].append(upper)
        parameters["kp"].append(stiffness)
        parameters["kd"].append(2.0 * 2.0 * armature * natural_frequency)
        parameters["effort_limit"].append(effort)
        parameters["velocity_limit"].append(velocity)
        parameters["action_scale"].append(0.25 * effort / stiffness)
    return parameters


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(directory: str | Path) -> dict[str, Any]:
    """Validate fixed filenames, dimensions, metadata, and every payload hash."""
    root = Path(directory)
    with (root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    fixed_values = {
        "schema_version": SCHEMA_VERSION, "control_dt": CONTROL_DT,
        "observation_dim": OBSERVATION_DIM, "action_dim": ACTION_DIM,
        "policy_file": POLICY_FILE, "reference_file": REFERENCE_FILE, "goals_file": GOALS_FILE,
    }
    for name, expected in fixed_values.items():
        if manifest.get(name) != expected:
            raise ValueError(f"manifest {name} must equal {expected!r}")
    for name in ("history_length", "goal_frames", "frame_count"):
        positive_integer(manifest.get(name), name)
    if manifest.get("goal_dim") != GOAL_FRAME_DIM * manifest["goal_frames"]:
        raise ValueError("manifest goal_dim must equal 73 * goal_frames")
    if manifest.get("joint_names") != G1_JOINT_NAMES:
        raise ValueError("manifest joint_names must use the 29-joint SDK order")
    for name in ("default_joint_position", "action_scale", "kp", "kd", "effort_limit", "velocity_limit", "joint_lower", "joint_upper"):
        values = np.asarray(manifest.get(name), dtype=np.float64)
        if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
            raise ValueError(f"manifest {name} must contain 29 finite values")
    for name in ("action_scale", "kp", "kd", "effort_limit", "velocity_limit"):
        if (np.asarray(manifest[name]) <= 0).any():
            raise ValueError(f"manifest {name} must be positive")
    if (np.asarray(manifest["joint_lower"]) >= np.asarray(manifest["joint_upper"])).any():
        raise ValueError("manifest lower limits must be less than upper limits")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(BUNDLE_FILES):
        raise ValueError(f"sha256 must contain exactly these fixed filenames: {BUNDLE_FILES}")
    for filename in BUNDLE_FILES:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle payload must be a regular file: {filename}")
        if hashes[filename] != sha256_file(path):
            raise ValueError(f"SHA256 mismatch for {filename}")
    for filename, width in ((REFERENCE_FILE, REFERENCE_DIM), (GOALS_FILE, manifest["goal_dim"])):
        if (root / filename).stat().st_size != manifest["frame_count"] * width * 4:
            raise ValueError(f"incorrect binary dimensions for {filename}")
        rows = np.fromfile(root / filename, dtype="<f4")
        if not np.isfinite(rows).all():
            raise ValueError(f"nonfinite binary values in {filename}")
    return manifest
