"""Export a single checkpoint and motion into a self-contained offline bundle.

Usage::

    python -m deployment.export_policy --checkpoint run/ckpt.pt \
        --motion motion.csv --mjcf data/g1_description/g1_29dof.xml \
        --output exported_policy

Only the checkpoint's ``pi`` is used. No world model, critic, simulator, FK,
optimizer state, or Isaac Lab module is instantiated. Export is CPU float32
even if training used bf16 autocast. The destination must not already exist.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp

from .contract import (
    ACTION_DIM, BUNDLE_FILES, CONTROL_DT, GOAL_FRAME_DIM, GOALS_FILE,
    OBSERVATION_DIM, PARITY_FILE, POLICY_FILE, REFERENCE_FILE, SCHEMA_VERSION,
    build_goals, build_observation, initialize_history, load_reference,
    positive_integer, robot_parameters, sha256_file, update_history, verify_bundle,
)


class RunningNorm(nn.Module):
    """Frozen training normalizer with identical state-dict keys and epsilon."""

    def __init__(self, dimension: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("var", torch.ones(dimension))
        self.register_buffer("count", torch.zeros(()))
        self.eps = 0.01

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / torch.sqrt(self.var + self.eps**2)


def _make_mlp(input_dimension: int, output_dimension: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dimension = input_dimension
    for hidden_dimension in (2048, 1024, 512, 256):
        layers.extend((nn.Linear(previous_dimension, hidden_dimension), nn.ELU()))
        previous_dimension = hidden_dimension
    layers.append(nn.Linear(previous_dimension, output_dimension))
    return nn.Sequential(*layers)


class Policy(nn.Module):
    """Inference-only copy of networks.Policy; exact enc/dec/norm key layout."""

    def __init__(self, history_length: int, goal_frames: int):
        super().__init__()
        self.hist = positive_integer(history_length, "history_length")
        self.goal_frames = positive_integer(goal_frames, "goal_frames")
        self.enc = _make_mlp(GOAL_FRAME_DIM * self.goal_frames, 32)
        self.dec = _make_mlp(32 + OBSERVATION_DIM * self.hist, ACTION_DIM)
        self.obs_norm = RunningNorm(OBSERVATION_DIM)
        self.goal_norm = RunningNorm(GOAL_FRAME_DIM * self.goal_frames)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(self.enc(self.goal_norm(goal)))
        observations = self.obs_norm(obs_hist).flatten(-2)
        return self.dec(torch.cat((latent, observations), dim=-1))


def load_policy(checkpoint: str | Path) -> tuple[Policy, dict[str, Any]]:
    """Securely load a strict pi state dict, rejecting uninitialized statistics."""
    try:
        contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError("checkpoint cannot be loaded safely with weights_only=True; only tensors and primitives are accepted") from error
    if not isinstance(contents, dict) or not isinstance(contents.get("pi"), Mapping):
        raise ValueError("checkpoint must contain a 'pi' state dict; no fallback to other weights is allowed")
    arguments = contents.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("checkpoint must contain an 'args' dictionary")
    history_length = positive_integer(arguments.get("hist"), "checkpoint args.hist")
    goal_frames = positive_integer(arguments.get("goal_frames"), "checkpoint args.goal_frames")
    state = contents["pi"]
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"pi state {name!r} must be a floating-point tensor")
        if not torch.isfinite(tensor).all() or not torch.isfinite(tensor.float()).all():
            raise ValueError(f"pi state {name!r} contains nonfinite float32 values")
    for prefix, dimension in (("obs_norm", OBSERVATION_DIM), ("goal_norm", GOAL_FRAME_DIM * goal_frames)):
        for suffix in ("mean", "var", "count"):
            name = f"{prefix}.{suffix}"
            tensor = state.get(name)
            expected_shape = () if suffix == "count" else (dimension,)
            if tensor is None or tuple(tensor.shape) != expected_shape:
                raise ValueError(f"pi {name} must have shape {expected_shape}")
            if suffix == "var" and (tensor < 0).any():
                raise ValueError(f"pi {name} must be nonnegative")
            if suffix == "count" and tensor.item() <= 0:
                raise ValueError(f"pi {name} must be positive; uninitialized normalization cannot be exported")
    policy = Policy(history_length, goal_frames).float().eval()
    policy.load_state_dict(state, strict=True)
    policy.requires_grad_(False)
    metadata = {
        "history_length": history_length,
        "goal_frames": goal_frames,
        "normalization_epsilon": 0.01,
        "normalization_count": {
            "observation": policy.obs_norm.count.item(), "goal": policy.goal_norm.count.item(),
        },
        "training_amp": str(arguments.get("amp", "none")),
    }
    return policy, metadata


@torch.inference_mode()
def build_parity_samples(
    policy: Policy,
    reference: np.ndarray,
    goals: np.ndarray,
    default_joint_position: list[float],
    sample_count: int = 8,
) -> dict[str, Any]:
    """Create deterministic, nontrivially rotated controller conformance cases.

    Histories use the same repeat-on-reset and oldest-first shift as training.
    Samples form one synthetic observation stream, visiting both clip ends.
    """
    count = positive_integer(sample_count, "parity sample count")
    generator = torch.Generator(device="cpu").manual_seed(20261026)
    frames = np.linspace(0, reference.shape[0] - 1, count, dtype=np.int64)
    history = None
    previous_action = 0.15 * torch.randn(ACTION_DIM, generator=generator)
    samples = []
    for frame in frames:
        row = torch.from_numpy(reference[frame])
        reference_rotation = quat_to_mat(row[3:7])
        rotation_offset = 0.25 * torch.randn(3, generator=generator) + torch.tensor([0.17, -0.23, 0.31])
        rotation = reference_rotation @ so3_exp(rotation_offset)
        quaternion = mat_to_quat(rotation)
        position = row[7:36] + 0.08 * torch.randn(ACTION_DIM, generator=generator)
        velocity = row[36:65] + 0.3 * torch.randn(ACTION_DIM, generator=generator)
        angular_velocity = rotation.T @ row[68:71] + 0.2 * torch.randn(3, generator=generator)
        observation = build_observation(
            position, velocity, quaternion, angular_velocity, previous_action,
            row[3:7], default_joint_position,
        )
        history = initialize_history(observation, policy.hist) if history is None else update_history(history, observation)
        goal = torch.from_numpy(goals[frame])
        action = policy(history.unsqueeze(0), goal.unsqueeze(0)).squeeze(0)
        if not torch.isfinite(action).all():
            raise ValueError("policy produced nonfinite parity actions")
        samples.append({
            "joint_position": position.tolist(),
            "joint_velocity": velocity.tolist(),
            "root_quaternion_wxyz": quaternion.tolist(),
            "angular_velocity_body": angular_velocity.tolist(),
            "previous_action": previous_action.tolist(),
            "frame": int(frame),
            "observation": observation.tolist(),
            "goal": goal.tolist(),
            "history": history.flatten().tolist(),
            "action": action.tolist(),
        })
        previous_action = action
    return {"samples": samples}


def export_onnx(policy: Policy, destination: str | Path) -> None:
    """Export a fixed-batch float32 graph with embedded checkpoint buffers."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX export requires the optional 'onnx' package") from error
    with torch.inference_mode():
        torch.onnx.export(
            policy,
            (
                torch.zeros(1, policy.hist, OBSERVATION_DIM, dtype=torch.float32),
                torch.zeros(1, GOAL_FRAME_DIM * policy.goal_frames, dtype=torch.float32),
            ),
            str(destination),
            input_names=["obs_hist", "goal"], output_names=["action"],
            opset_version=17, dynamo=False,
            # Keep normalizer buffers visible rather than baking their values
            # into derived scale/bias initializers.
            do_constant_folding=False,
            external_data=False,
        )
    graph = onnx.load(str(destination), load_external_data=False)
    onnx.checker.check_model(graph)
    for tensors, specification in (
        (graph.graph.input, {"obs_hist": [1, policy.hist, OBSERVATION_DIM], "goal": [1, GOAL_FRAME_DIM * policy.goal_frames]}),
        (graph.graph.output, {"action": [1, ACTION_DIM]}),
    ):
        if {tensor.name for tensor in tensors} != set(specification):
            raise ValueError("exported ONNX graph has incorrect input/output names")
        for tensor in tensors:
            tensor_type = tensor.type.tensor_type
            dimensions = [dimension.dim_value for dimension in tensor_type.shape.dim]
            if tensor_type.elem_type != onnx.TensorProto.FLOAT or dimensions != specification[tensor.name]:
                raise ValueError(f"exported ONNX tensor {tensor.name!r} has incorrect float32 shape")
    if any(initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in graph.graph.initializer):
        raise ValueError("policy.onnx must not refer to external tensor files")


def validate_onnx_parity(path: str | Path, parity: dict[str, Any], history_length: int) -> dict[str, Any]:
    """Check ONNX Runtime against torch when that optional verifier is installed."""
    try:
        import onnxruntime
    except ImportError:
        return {"onnxruntime_checked": False, "reason": "onnxruntime is not installed"}
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    maximum_error = 0.0
    for sample in parity["samples"]:
        action = session.run(["action"], {
            "obs_hist": np.asarray(sample["history"], dtype=np.float32).reshape(1, history_length, OBSERVATION_DIM),
            "goal": np.asarray(sample["goal"], dtype=np.float32)[None],
        })[0]
        expected = np.asarray(sample["action"], dtype=np.float32)[None]
        if not np.isfinite(action).all() or not np.allclose(action, expected, rtol=1e-4, atol=1e-5):
            raise ValueError("ONNX Runtime parity failed (rtol=1e-4, atol=1e-5)")
        maximum_error = max(maximum_error, float(np.max(np.abs(action - expected))))
    return {"onnxruntime_checked": True, "maximum_absolute_action_error": maximum_error}


def _source_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    source_files = (
        "deployment/contract.py", "deployment/export_policy.py", "supermimic/networks.py",
        "supermimic/motion.py", "supermimic/mdp.py", "supermimic/robot.py", "supermimic/rotations.py",
    )
    provenance: dict[str, Any] = {
        "source_sha256": {name: sha256_file(repository / name) for name in source_files},
    }
    try:
        provenance["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, text=True, capture_output=True, check=True,
        ).stdout
        provenance["git_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        provenance.update(git_commit=None, git_dirty=None)
    return provenance


def _write_json(path: Path, contents: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(contents, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _publish_directory(temporary: Path, destination: Path) -> None:
    """Publish in one rename; Linux RENAME_NOREPLACE also closes the race."""
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists; refusing to overwrite: {destination}")
    if os.name == "posix":
        library = ctypes.CDLL(None, use_errno=True)
        rename_no_replace = getattr(library, "renameat2", None)
        if rename_no_replace is not None:
            rename_no_replace.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename_no_replace.restype = ctypes.c_int
            result = rename_no_replace(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1)
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                raise OSError(error_number, os.strerror(error_number), str(destination))
    # os.rename never overwrites a nonempty bundle directory. The precheck also
    # protects empty preexisting destinations on platforms lacking renameat2.
    os.rename(temporary, destination)


def export_bundle(
    checkpoint: str | Path,
    motion: str | Path,
    output: str | Path,
    mjcf: str | Path,
    *,
    src_fps: float = 30.0,
    parity_samples: int = 8,
) -> dict[str, Any]:
    """Validate everything in a sibling temporary directory, then publish it."""
    # Do not resolve the final component: a dangling output symlink also counts
    # as an existing destination and must not redirect the export elsewhere.
    destination = Path(os.path.abspath(output))
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists; refusing to overwrite: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {destination.parent}")
    sample_count = positive_integer(parity_samples, "parity_samples")
    sources = {"checkpoint": Path(checkpoint).resolve(), "motion": Path(motion).resolve(), "mjcf": Path(mjcf).resolve()}
    source_metadata = {name: {"source": str(path), "sha256": sha256_file(path)} for name, path in sources.items()}
    policy, policy_metadata = load_policy(sources["checkpoint"])
    parameters = robot_parameters(sources["mjcf"])
    reference = load_reference(sources["motion"], src_fps=src_fps)
    goals = build_goals(reference, policy.goal_frames)
    parity = build_parity_samples(policy, reference, goals, parameters["default_joint_position"], sample_count)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary_directory:
        temporary = Path(temporary_directory)
        export_onnx(policy, temporary / POLICY_FILE)
        reference.astype("<f4").tofile(temporary / REFERENCE_FILE)
        goals.astype("<f4").tofile(temporary / GOALS_FILE)
        _write_json(temporary / PARITY_FILE, parity)
        parity_metadata = validate_onnx_parity(temporary / POLICY_FILE, parity, policy.hist)
        # Hash before and after loading, so an input changed mid-export is not
        # silently attributed to the wrong checkpoint or motion.
        for name, path in sources.items():
            if source_metadata[name]["sha256"] != sha256_file(path):
                raise ValueError(f"input {name} changed during export")
        manifest = {
            "schema_version": SCHEMA_VERSION, "control_dt": CONTROL_DT,
            "history_length": policy.hist, "goal_frames": policy.goal_frames,
            "observation_dim": OBSERVATION_DIM, "goal_dim": GOAL_FRAME_DIM * policy.goal_frames,
            "action_dim": ACTION_DIM, "frame_count": reference.shape[0],
            **parameters,
            "policy_file": POLICY_FILE, "reference_file": REFERENCE_FILE, "goals_file": GOALS_FILE,
            "sha256": {name: sha256_file(temporary / name) for name in BUNDLE_FILES},
            "metadata": {
                **source_metadata, **_source_provenance(), **policy_metadata, **parity_metadata,
                "source_fps": src_fps, "export_dtype": "float32", "torch_version": str(torch.__version__),
                "reference_layout": [
                    "root_position_world:3", "root_quaternion_wxyz:4", "joint_position:29",
                    "joint_velocity:29", "root_linear_velocity_world:3", "root_angular_velocity_world:3",
                ],
                "rotation_6d": "first two matrix columns, row-major interleaved",
                "history_order": "oldest_to_newest; repeat first observation on reset",
                "parity_sample_count": sample_count,
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        verify_bundle(temporary)
        _publish_directory(temporary, destination)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="training checkpoint with pi and args")
    parser.add_argument("--motion", required=True, type=Path, help="one LAFAN retarget CSV (xyzw input quaternion)")
    parser.add_argument("--output", required=True, type=Path, help="fresh bundle directory; parent must exist")
    parser.add_argument("--mjcf", required=True, type=Path, help="G1 MJCF used to compile the 29 hard joint ranges")
    parser.add_argument("--src-fps", type=float, default=30.0, help="motion source rate (default: 30 Hz)")
    parser.add_argument("--parity", "--parity-samples", dest="parity_samples", type=int, default=8,
                        help="number of deterministic parity cases (default: 8; must be positive)")
    arguments = parser.parse_args(argv)
    try:
        manifest = export_bundle(**vars(arguments))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"export failed: {error}\n")
    print(f"Exported {manifest['frame_count']} frames to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
