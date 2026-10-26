"""Headless MuJoCo bridge for isolated Unitree HG SDK2 deployment checks.

This module never imports the training stack. SDK2 and MuJoCo are imported only
by the code paths that need them, so the contract and safety helpers work without
either package. Run ``python deployment/sim_bridge.py --help`` for the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


LOGGER = logging.getLogger("supermimic.sim_bridge")
JOINT_COUNT = 29
REFERENCE_WIDTH = 71
PHYSICS_DT = 0.002
CONTROL_DT = 0.02
# Mirror the version-1 wire contract without importing contract.py's torch stack.
BUNDLE_FILES = ("policy.onnx", "reference.bin", "goals.bin", "parity.json")
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def validate_dds_isolation(interface: str, domain: int) -> None:
    """Fail closed before importing SDK2 or initializing any DDS participant."""
    if interface != "lo":
        raise ValueError("Simulation DDS must bind to interface 'lo', never a robot interface")
    if isinstance(domain, bool) or not isinstance(domain, int) or not 1 <= domain <= 232:
        raise ValueError("Simulation DDS requires a nonzero domain in [1, 232] (default 42)")
    # Even apparently safe XML can contain includes or a competing interface rule.
    if "CYCLONEDDS_URI" in os.environ:
        raise ValueError("Unset CYCLONEDDS_URI; this bridge refuses all DDS configuration overrides")


def _joint_vector(manifest: dict[str, Any], name: str) -> np.ndarray:
    vector = np.asarray(manifest.get(name), dtype=np.float64)
    if vector.shape != (JOINT_COUNT,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"manifest.{name} must contain 29 finite numbers")
    return vector


@dataclass(frozen=True)
class SimulationBundle:
    directory: Path
    joint_names: tuple[str, ...]
    default_joint_position: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    velocity_limit: np.ndarray
    action_scale: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    reference: np.ndarray

    @classmethod
    def load(cls, directory: str | Path) -> SimulationBundle:
        directory = Path(directory).resolve()
        with (directory / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        fixed_values = {
            "schema_version": 1, "control_dt": CONTROL_DT,
            "observation_dim": 99, "action_dim": JOINT_COUNT,
            "policy_file": "policy.onnx", "reference_file": "reference.bin", "goals_file": "goals.bin",
        }
        for name, expected in fixed_values.items():
            if manifest.get(name) != expected:
                raise ValueError(f"manifest.{name} must equal {expected!r}")
        for name in ("history_length", "goal_frames", "frame_count"):
            value = manifest.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"manifest.{name} must be a positive integer")
        if manifest.get("goal_dim") != 73 * manifest["goal_frames"]:
            raise ValueError("manifest.goal_dim must equal 73 * goal_frames")
        names = manifest.get("joint_names")
        if names != list(G1_JOINT_NAMES):
            raise ValueError("manifest.joint_names must use the 29-joint SDK order")
        frame_count = manifest["frame_count"]
        vectors = {name: _joint_vector(manifest, name) for name in (
            "default_joint_position", "kp", "kd", "effort_limit", "velocity_limit",
            "action_scale", "joint_lower", "joint_upper",
        )}
        for name in ("kp", "kd", "effort_limit", "velocity_limit", "action_scale"):
            if np.any(vectors[name] <= 0):
                raise ValueError(f"manifest.{name} must be positive")
        if np.any(vectors["joint_lower"] >= vectors["joint_upper"]):
            raise ValueError("manifest joint_lower must be less than joint_upper")
        hashes = manifest.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(BUNDLE_FILES):
            raise ValueError(f"sha256 must contain exactly these fixed filenames: {BUNDLE_FILES}")
        for filename in BUNDLE_FILES:
            payload_path = directory / filename
            if payload_path.is_symlink() or not payload_path.is_file():
                raise ValueError(f"bundle payload must be a regular file: {filename}")
            digest = hashlib.sha256()
            with payload_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if hashes[filename] != digest.hexdigest():
                raise ValueError(f"SHA256 mismatch for {filename}")
        reference_path = directory / "reference.bin"
        if reference_path.stat().st_size != frame_count * REFERENCE_WIDTH * 4:
            raise ValueError("reference.bin size must equal frame_count * 71 * 4 bytes")
        reference = np.fromfile(reference_path, dtype="<f4").reshape(frame_count, REFERENCE_WIDTH)
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference.bin contains nonfinite values")
        quaternion_norms = np.linalg.norm(reference[:, 3:7], axis=1)
        if not np.allclose(quaternion_norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError("reference.bin root quaternions must be normalized wxyz")
        goals_path = directory / "goals.bin"
        if goals_path.stat().st_size != frame_count * manifest["goal_dim"] * 4:
            raise ValueError("incorrect binary dimensions for goals.bin")
        if not np.all(np.isfinite(np.fromfile(goals_path, dtype="<f4"))):
            raise ValueError("nonfinite binary values in goals.bin")
        return cls(directory=directory, joint_names=tuple(names), reference=reference, **vectors)


@dataclass(frozen=True)
class JointMapping:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    root_joint_id: int
    root_body_id: int
    root_qpos_address: int
    root_dof_address: int


def map_joints(model: Any, joint_names: tuple[str, ...]) -> JointMapping:
    import mujoco

    if len(joint_names) != JOINT_COUNT or len(set(joint_names)) != JOINT_COUNT:
        raise ValueError("Expected 29 unique named robot joints")
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError("MJCF must have exactly one freejoint for the robot root")
    root_joint_id = int(free_joints[0])
    root_body_id = int(model.jnt_bodyid[root_joint_id])
    joint_ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Missing manifest joint in MJCF: {name}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"Manifest joint must have scalar qpos and velocity: {name}")
        body_id = int(model.jnt_bodyid[joint_id])
        while body_id and body_id != root_body_id:
            body_id = int(model.body_parentid[body_id])
        if body_id != root_body_id:
            raise ValueError(f"Manifest joint does not descend from root freejoint: {name}")
        joint_ids.append(joint_id)
    joint_ids = np.asarray(joint_ids, dtype=np.int32)
    return JointMapping(
        joint_ids=joint_ids,
        qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids]).copy(),
        dof_addresses=np.asarray(model.jnt_dofadr[joint_ids]).copy(),
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        root_qpos_address=int(model.jnt_qposadr[root_joint_id]),
        root_dof_address=int(model.jnt_dofadr[root_joint_id]),
    )


def build_model(mjcf: str | Path, bundle: SimulationBundle) -> tuple[Any, Any, JointMapping]:
    """Load the original path with MjSpec, retaining compiler/include asset bases."""
    import mujoco

    specification = mujoco.MjSpec.from_file(str(Path(mjcf).resolve()))
    # A native position actuator with ctrl=0 still exerts force. Delete all native
    # actuators so qfrc_applied below is the only commanded motor torque.
    for actuator in list(specification.actuators):
        specification.delete(actuator)
    specification.worldbody.add_geom(
        name="supermimic_bridge_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1], rgba=[0.25, 0.3, 0.35, 1], contype=2, conaffinity=1,
    )
    free_joints = [joint for joint in specification.joints
                   if int(joint.type) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if len(free_joints) != 1:
        raise ValueError("MJCF must have exactly one freejoint for the robot root")
    root_body = free_joints[0].parent
    world_body = specification.worldbody
    # Set masks before compiling: changing geom masks alone leaves body masks
    # and collision acceleration structures inconsistent for visual-only geoms.
    for geometry in specification.geoms:
        ancestor = geometry.parent
        # MuJoCo 3.11's worldbody.parent is not a safe/null sentinel. Never
        # dereference it, even when classifying unrelated environment geometry.
        while ancestor is not None and ancestor is not root_body and ancestor is not world_body:
            ancestor = ancestor.parent
        geometry.contype = 1 if ancestor is root_body else 2
        geometry.conaffinity = 2 if ancestor is root_body else 1
    model = specification.compile()
    mapping = map_joints(model, bundle.joint_names)
    model.opt.timestep = PHYSICS_DT
    model.opt.gravity[:] = [0, 0, -9.81]
    model.dof_armature[mapping.dof_addresses] = bundle.kp / (20 * math.pi) ** 2
    robot_bodies = np.zeros(model.nbody, dtype=bool)
    for body_id in range(1, model.nbody):
        robot_bodies[body_id] = (
            body_id == mapping.root_body_id or robot_bodies[model.body_parentid[body_id]]
        )
    robot_geoms = robot_bodies[model.geom_bodyid]
    if model.npair and np.any(robot_geoms[model.pair_geom1] & robot_geoms[model.pair_geom2]):
        raise ValueError("Explicit robot-robot contact pairs bypass self-collision masks; remove them")
    data = mujoco.MjData(model)
    reset_from_reference(model, data, mapping, bundle.reference[0])
    return model, data, mapping


def reset_from_reference(model: Any, data: Any, mapping: JointMapping, frame: np.ndarray) -> None:
    import mujoco

    mujoco.mj_resetData(model, data)
    root_position = mapping.root_qpos_address
    root_velocity = mapping.root_dof_address
    data.qpos[root_position:root_position + 7] = frame[:7]
    data.qpos[root_position + 3:root_position + 7] /= np.linalg.norm(frame[3:7])
    data.qpos[mapping.qpos_addresses] = frame[7:36]
    data.qvel[mapping.dof_addresses] = frame[36:65]
    data.qvel[root_velocity:root_velocity + 3] = frame[65:68]
    mujoco.mj_forward(model, data)
    # Freejoint linear qvel is world-frame; rotational qvel is body-frame.
    root_rotation = data.xmat[mapping.root_body_id].reshape(3, 3)
    data.qvel[root_velocity + 3:root_velocity + 6] = root_rotation.T @ frame[68:71]
    mujoco.mj_forward(model, data)


@dataclass(frozen=True)
class MotorCommand:
    position: np.ndarray
    velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque: np.ndarray
    received_at: float


def decode_command(message: Any, bundle: SimulationBundle, received_at: float) -> MotorCommand:
    if message.mode_pr != 0:
        raise ValueError("Only HG mode_pr=0 (PR joint commands) is supported")
    if len(message.motor_cmd) < JOINT_COUNT:
        raise ValueError("HG command has fewer than 29 motors")
    values = np.asarray([
        [motor.q, motor.dq, motor.kp, motor.kd, motor.tau]
        for motor in message.motor_cmd[:JOINT_COUNT]
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)) or not math.isfinite(received_at):
        raise ValueError("Motor targets, gains, torque and receipt time must be finite")
    if np.any(values[:, 2:4] < 0):
        raise ValueError("Motor gains must be nonnegative")
    modes = np.asarray([motor.mode for motor in message.motor_cmd[:JOINT_COUNT]])
    if not np.all((modes == 0) | (modes == 1)):
        raise ValueError("Only motor mode 0 (disabled) or 1 (enabled) is supported")
    enabled = modes == 1
    return MotorCommand(
        position=np.clip(values[:, 0], bundle.joint_lower, bundle.joint_upper),
        velocity=np.clip(values[:, 1], -bundle.velocity_limit, bundle.velocity_limit),
        kp=np.minimum(values[:, 2], bundle.kp) * enabled,
        kd=np.minimum(values[:, 3], bundle.kd) * enabled,
        torque=np.clip(values[:, 4], -bundle.effort_limit, bundle.effort_limit) * enabled,
        received_at=received_at,
    )


def compute_torque(
    bundle: SimulationBundle, command: MotorCommand | None, position: np.ndarray,
    velocity: np.ndarray, now: float, timeout: float,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Return final effort-limited torque, watchdog flag and pre-limit torque."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be finite and positive")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise FloatingPointError("Nonfinite simulation state")
    stale = command is None or not 0 <= now - command.received_at <= timeout
    if stale:
        requested = -bundle.kd * velocity
    else:
        requested = (command.kp * (command.position - position)
                     + command.kd * (command.velocity - velocity) + command.torque)
    applied = np.clip(requested, -bundle.effort_limit, bundle.effort_limit)
    if not np.all(np.isfinite(applied)):
        raise FloatingPointError("Nonfinite applied torque")
    return applied, stale, requested


def body_angular_velocity(model: Any, data: Any, body_id: int) -> np.ndarray:
    import mujoco

    spatial_velocity = np.zeros(6, dtype=np.float64)
    # XBODY is the body frame; BODY uses the inertial/principal-axis frame.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, spatial_velocity, 1)
    # MuJoCo spatial vectors are ANGULAR then LINEAR, not the reverse.
    return spatial_velocity[:3].copy()


class SimulationBridge:
    """A single simulation thread plus a lock-protected, validated DDS mailbox."""

    def __init__(self, bundle: SimulationBundle, model: Any, data: Any, mapping: JointMapping,
                 *, command_timeout: float = 0.1, freeze_before_command: bool = False,
                 mode_machine: int = 5, drop_state_after: float | None = None,
                 drop_command_after: float | None = None) -> None:
        if not math.isfinite(command_timeout) or command_timeout <= 0:
            raise ValueError("command_timeout must be finite and positive")
        for threshold in (drop_state_after, drop_command_after):
            if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
                raise ValueError("Fault injection thresholds must be finite and nonnegative")
        if not 0 <= mode_machine <= 255:
            raise ValueError("mode_machine must fit uint8")
        self.bundle, self.model, self.data, self.mapping = bundle, model, data, mapping
        self.command_timeout = command_timeout
        self.freeze_before_command = freeze_before_command
        self.mode_machine = mode_machine
        self.drop_state_after = drop_state_after
        self.drop_command_after = drop_command_after
        self.lock = threading.Lock()
        self.command: MotorCommand | None = None
        self.started_at = time.monotonic()
        self.accepted_commands = 0
        self.rejected_commands = 0
        self.dropped_commands = 0
        self.command_intervals: list[float] = []
        self.last_received_at: float | None = None
        self.published_states = 0
        self.tick = 0
        self.publisher = None
        self.subscriber = None
        self.low_state = None
        self.crc = None

    def connect(self, interface: str = "lo", domain: int = 42) -> None:
        validate_dds_isolation(interface, domain)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain, interface)
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.crc = CRC()
        self.publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.subscriber.Init(self.receive_command, 1)

    def receive_command(self, message: Any) -> None:
        received_at = time.monotonic()
        with self.lock:
            if (self.drop_command_after is not None
                    and received_at - self.started_at >= self.drop_command_after):
                self.dropped_commands += 1
                return
            try:
                command = decode_command(message, self.bundle, received_at)
            except (ValueError, TypeError, AttributeError, OverflowError):
                self.command = None  # Invalid commands immediately disarm stale targets.
                self.rejected_commands += 1
                return
            if self.last_received_at is not None:
                self.command_intervals.append(received_at - self.last_received_at)
            self.last_received_at = received_at
            self.command = command
            self.accepted_commands += 1

    def publish_state(self) -> None:
        if self.publisher is None:
            raise RuntimeError("DDS is not initialized")
        state = self.low_state
        state.mode_pr = 0
        state.mode_machine = self.mode_machine
        state.tick = self.tick
        for motor_index, motor in enumerate(state.motor_state):
            motor.mode = 1 if motor_index < JOINT_COUNT else 0
            if motor_index < JOINT_COUNT:
                motor.q = float(self.data.qpos[self.mapping.qpos_addresses[motor_index]])
                motor.dq = float(self.data.qvel[self.mapping.dof_addresses[motor_index]])
                motor.ddq = float(self.data.qacc[self.mapping.dof_addresses[motor_index]])
                motor.tau_est = float(self.data.qfrc_applied[self.mapping.dof_addresses[motor_index]])
            else:
                motor.q = motor.dq = motor.ddq = motor.tau_est = 0.0
            if isinstance(motor.temperature, (int, float)):
                motor.temperature = 0
            else:
                motor.temperature[:] = [0] * len(motor.temperature)
        state.imu_state.quaternion[:] = self.data.xquat[self.mapping.root_body_id].tolist()
        state.imu_state.gyroscope[:] = body_angular_velocity(
            self.model, self.data, self.mapping.root_body_id,
        ).tolist()
        # No physical accelerometer/site calibration is assumed by this bridge.
        state.imu_state.accelerometer[:] = [0.0, 0.0, 0.0]
        state.crc = self.crc.Crc(state)
        self.publisher.Write(state)
        self.published_states += 1

    def run(self, seconds: float) -> dict[str, Any]:
        import mujoco

        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        self.started_at = time.monotonic()
        deadline = self.started_at
        end_at = self.started_at + seconds
        steps = frozen_steps = stale_steps = clipped_steps = clipped_motors = 0
        maximum_lateness = maximum_torque = 0.0
        position_squared_error = np.zeros(JOINT_COUNT)
        root_squared_error = np.zeros(3)
        orientation_squared_error = 0.0
        command_ages: list[float] = []
        final_applied = np.zeros(JOINT_COUNT)
        while True:
            now = time.monotonic()
            if now >= end_at:
                break
            if now < deadline:
                time.sleep(min(deadline - now, end_at - now))
                continue
            maximum_lateness = max(maximum_lateness, now - deadline)
            with self.lock:
                command = self.command
            frozen = self.freeze_before_command and self.accepted_commands == 0
            if frozen:
                frozen_steps += 1
                self.data.qfrc_applied[:] = 0
            else:
                applied, stale, requested = compute_torque(
                    self.bundle, command, self.data.qpos[self.mapping.qpos_addresses],
                    self.data.qvel[self.mapping.dof_addresses], now, self.command_timeout,
                )
                self.data.qfrc_applied[:] = 0
                self.data.qfrc_applied[self.mapping.dof_addresses] = applied
                final_applied = applied
                clipped = np.abs(requested) > self.bundle.effort_limit
                clipped_steps += int(np.any(clipped))
                clipped_motors += int(np.count_nonzero(clipped))
                maximum_torque = max(maximum_torque, float(np.max(np.abs(applied))))
                stale_steps += int(stale)
                if command is not None and not stale:
                    command_ages.append(now - command.received_at)
                previous_sim_time = float(self.data.time)
                mujoco.mj_step(self.model, self.data)
                if (self.data.time <= previous_sim_time or not np.all(np.isfinite(self.data.qpos))
                        or not np.all(np.isfinite(self.data.qvel))):
                    raise FloatingPointError("MuJoCo diverged or automatically reset simulation time")
                # mj_step leaves derived poses at the pre-integration position.
                mujoco.mj_forward(self.model, self.data)
                reference_index = min(
                    int((self.data.time + 1e-10) / CONTROL_DT), len(self.bundle.reference) - 1,
                )
                frame = self.bundle.reference[reference_index]
                position_squared_error += (self.data.qpos[self.mapping.qpos_addresses] - frame[7:36]) ** 2
                root_squared_error += (self.data.xpos[self.mapping.root_body_id] - frame[:3]) ** 2
                cosine_half_angle = abs(float(np.dot(self.data.xquat[self.mapping.root_body_id], frame[3:7])))
                orientation_squared_error += (2 * math.acos(min(1.0, cosine_half_angle))) ** 2
                steps += 1
            if self.drop_state_after is None or now - self.started_at < self.drop_state_after:
                if self.publisher is not None:
                    self.publish_state()
            # HG tick is milliseconds, increasing even while the startup pose is frozen.
            self.tick = (self.tick + int(round(PHYSICS_DT * 1000))) & 0xFFFFFFFF
            deadline += PHYSICS_DT
        with self.lock:
            accepted_commands = self.accepted_commands
            rejected_commands = self.rejected_commands
            dropped_commands = self.dropped_commands
            intervals = list(self.command_intervals)
        return {
            "physics_dt": PHYSICS_DT, "control_dt": CONTROL_DT,
            "wall_seconds": time.monotonic() - self.started_at,
            "simulation_seconds": float(self.data.time), "physics_steps": steps,
            "frozen_steps": frozen_steps, "stale_steps": stale_steps,
            "accepted_commands": accepted_commands, "rejected_commands": rejected_commands,
            "dropped_commands": dropped_commands, "published_states": self.published_states,
            "clipped_steps": clipped_steps, "clipped_motor_samples": clipped_motors,
            "maximum_applied_torque": maximum_torque,
            "maximum_schedule_lateness_seconds": maximum_lateness,
            "command_interarrival_seconds": summarize_samples(intervals),
            "command_receipt_to_step_seconds": summarize_samples(command_ages),
            "joint_position_rmse": np.sqrt(position_squared_error / steps).tolist() if steps else None,
            "root_position_rmse": np.sqrt(root_squared_error / steps).tolist() if steps else None,
            "root_orientation_rmse_radians": math.sqrt(orientation_squared_error / steps) if steps else None,
            "reference_exhausted": bool(self.data.time >= len(self.bundle.reference) * CONTROL_DT),
            "final_reference_frame": min(int(self.data.time / CONTROL_DT), len(self.bundle.reference) - 1),
            "final_joint_position": self.data.qpos[self.mapping.qpos_addresses].tolist(),
            "final_joint_velocity": self.data.qvel[self.mapping.dof_addresses].tolist(),
            "final_root_position": self.data.xpos[self.mapping.root_body_id].tolist(),
            "final_root_quaternion_wxyz": self.data.xquat[self.mapping.root_body_id].tolist(),
            "final_applied_torque": final_applied.tolist(),
            "tracking_frame_clock": "simulation time, final reference frame held after clip end",
            "success_evaluated": False,
        }

    def close(self) -> None:
        self.data.qfrc_applied[:] = 0
        for endpoint in (self.subscriber, self.publisher):
            if endpoint is not None:
                endpoint.Close()


def summarize_samples(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {"count": len(samples), "mean": float(np.mean(samples)),
            "p95": float(np.percentile(samples, 95)), "max": float(np.max(samples))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True, help="Generated robot XML; assets remain relative to it")
    parser.add_argument("--interface", default="lo", choices=["lo"])
    parser.add_argument("--domain", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=10.0, help="Wall-clock duration, not clip phase")
    parser.add_argument("--command-timeout", type=float, default=0.1)
    parser.add_argument("--freeze-before-command", action="store_true")
    parser.add_argument("--mode-machine", type=int, default=5)
    parser.add_argument("--drop-state-after", type=float, help="Stop publishing at this elapsed wall time")
    parser.add_argument("--drop-command-after", type=float, help="Ignore commands after this elapsed wall time")
    parser.add_argument("--metrics", type=Path, help="Optional JSON output; parent directory must exist")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    validate_dds_isolation(arguments.interface, arguments.domain)
    if not math.isfinite(arguments.seconds) or arguments.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    bundle = SimulationBundle.load(arguments.bundle)
    model, data, mapping = build_model(arguments.mjcf, bundle)
    bridge = SimulationBridge(
        bundle, model, data, mapping, command_timeout=arguments.command_timeout,
        freeze_before_command=arguments.freeze_before_command, mode_machine=arguments.mode_machine,
        drop_state_after=arguments.drop_state_after, drop_command_after=arguments.drop_command_after,
    )
    LOGGER.warning("ISOLATED SIMULATION ONLY: DDS interface=lo domain=%d", arguments.domain)
    LOGGER.warning("Robot self-collision DISABLED; robot/environment masks 1/2 and 2/1")
    LOGGER.warning("Explicit effort-clipped PD torque, not original position-only bridge semantics")
    LOGGER.warning("Physics/state writer 500 Hz (dt=.002), deliberately not original dt=.005")
    LOGGER.info("Reference frame 0 reset includes world linear/angular velocities; no GUI")
    try:
        bridge.connect(arguments.interface, arguments.domain)
        metrics = bridge.run(arguments.seconds)
    finally:
        bridge.close()
    encoded = json.dumps(metrics, indent=2, allow_nan=False)
    if arguments.metrics:
        arguments.metrics.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
