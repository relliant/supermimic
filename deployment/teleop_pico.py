"""Pico -> existing XRoboToolkit SDK -> external GMR -> local live reference.

``--check`` is hardware-free. ``--run`` requires an already-running SDK service
and an already-running *live* reference receiver; it starts neither service and
never connects to a robot. Right A toggles arm/pause, left X exits (with priority).
Every discontinuity requires a fresh A edge and a newly filled delay buffer.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

from .reference_stream import CONTROL_DT, DEFAULT_CHANNEL, connect_redis

DEFAULT_MJCF = Path(__file__).resolve().parents[1] / "data" / "g1_description" / "g1_29dof.xml"

SDK_GETTERS = (
    "init", "close", "is_body_data_available", "get_body_timestamp_ns",
    "get_body_joints_pose", "get_A_button", "get_X_button",
)


class SourceDiscontinuity(ValueError):
    """The body clock cannot safely continue the current session."""


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def external_paths(wbc_root: Path) -> dict[str, Path]:
    root = wbc_root.expanduser().resolve()
    package = root / "data_processing" / "general_motion_retargeting"
    paths = {
        "data_processing": package.parent,
        "xrobot_source": package / "xrobot_utils.py",
        "gmr_source": package / "motion_retarget.py",
        "model": package.parent / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
        "config": package / "ik_configs" / "xrobot_to_g1.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete --wbc-root: " + ", ".join(missing))
    return paths


def _provenance(paths: dict[str, Path]) -> dict:
    return {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items() if path.is_file()
    }


def check_environment(wbc_root: Path, mjcf: Path = DEFAULT_MJCF) -> dict:
    """Inspect specs/metadata and files, never import SDK/GMR or initialize SDK."""
    paths = external_paths(wbc_root)
    paths["simulator_mjcf"] = mjcf.expanduser().resolve(strict=True)
    modules = {}
    distributions = importlib.metadata.packages_distributions()
    for module_name in ("xrobotoolkit_sdk", "mink", "daqp", "mujoco", "scipy", "cv2", "rich",
                        "redis", "torch", "imageio", "loop_rate_limiters"):
        try:
            specification = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            specification = None
        versions = {}
        for distribution in distributions.get(module_name, []):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        modules[module_name] = {
            "available": specification is not None,
            "origin": None if specification is None else specification.origin,
            "distributions": versions,
        }
    return {
        "interpreter": sys.executable,
        "python": sys.version,
        "modules": modules,
        "external": _provenance(paths),
        "sdk_required_getters": list(SDK_GETTERS),
        "sdk_getter_validation": "runtime only; --check does not import or initialize SDK",
        "simulator_limit_validation": "runtime only; --check hashes MJCF without compiling it",
        "sdk_body_clock": (
            "body timestamp, or a pinned positive body-joint timestamp when the aggregate is zero; "
            "nanoseconds only, never get_time_stamp_ns"
        ),
        "gmr": {"src_human": "xrobot", "tgt_robot": "unitree_g1", "solver": "daqp",
                "joint_units": "radians", "offset_to_ground": True, "grounding_offset_m": 0.1},
        "hardware_initialized": False,
    }


class BodyClock:
    """Map arbitrary-origin SDK nanoseconds to one Unix acquisition timeline."""

    def __init__(self, max_gap: float = 0.15):
        self.max_gap = max_gap
        self.reset()

    def reset(self) -> None:
        self.anchor_sequence: int | None = None
        self.anchor_unix = 0.0
        self.anchor_monotonic = 0.0
        self.previous_sequence: int | None = None
        self.previous_monotonic: float | None = None
        self.mapped_time = 0.0

    def map(self, sequence: int, *, unix_time: float, monotonic_time: float) -> float | None:
        if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)) or sequence <= 0:
            raise SourceDiscontinuity(
                f"SDK body timestamp must be positive integer nanoseconds; got {sequence!r} ({type(sequence).__name__})"
            )
        sequence = int(sequence)
        delta = 0.0
        if self.previous_sequence is not None:
            delta = (sequence - self.previous_sequence) * 1e-9
            if delta == 0:
                return None
            if delta < 0:
                raise SourceDiscontinuity("SDK body timestamp regressed")
            if delta < 1e-5:
                # Coalesce tiny positive updates instead of interpreting them
                # as a unit error. Keep the last accepted anchor unchanged so
                # neither a duplicate nor timestamp noise refreshes freshness.
                return None
            if delta > self.max_gap:
                raise SourceDiscontinuity("SDK body timestamp gap exceeded max_gap")
        if self.anchor_sequence is None:
            self.anchor_sequence = sequence
            self.anchor_unix = unix_time
            self.anchor_monotonic = monotonic_time
            self.mapped_time = unix_time
        else:
            receipt_dt = monotonic_time - self.previous_monotonic
            if receipt_dt < 0 or abs(delta - receipt_dt) > 0.5:
                raise SourceDiscontinuity("SDK body nanosecond clock diverged from acquisition clock")
            predicted = self.mapped_time + delta
            host_acquisition_time = self.anchor_unix + monotonic_time - self.anchor_monotonic
            # Slew at <=1000 ppm on *new body samples only*. This tracks normal
            # device/host oscillator drift without stamping old samples fresh,
            # resetting the interpolation timeline, or following wall-clock jumps.
            correction = float(np.clip((host_acquisition_time - predicted) * .01, -delta * .001, delta * .001))
            self.mapped_time = predicted + correction
        self.previous_sequence = sequence
        self.previous_monotonic = monotonic_time
        return self.mapped_time


@dataclass(frozen=True)
class BodySample:
    source_sequence: int
    source_time: float
    received_monotonic: float
    body: dict


class XRoboSource:
    """Read body snapshots without the SDK's unrelated global/controller clock."""

    def __init__(self, sdk, streamer):
        self.sdk = sdk
        self.streamer = streamer
        missing = [name for name in SDK_GETTERS if not callable(getattr(sdk, name, None))]
        if missing:
            raise RuntimeError("XRoboToolkit SDK missing required API: " + ", ".join(missing))
        self.clock_source: str | None = None
        self.clock_joint_index: int | None = None
        self.ignore_sdk_timestamps = False
        self._last_host_sample = -math.inf

    def _joint_timestamps(self) -> np.ndarray:
        getter = getattr(self.sdk, "get_body_joints_timestamp", None)
        if not callable(getter):
            raise SourceDiscontinuity(
                "SDK body timestamp is zero and get_body_joints_timestamp is unavailable; "
                "a body-specific acquisition clock is required"
            )
        timestamps = np.asarray(getter())
        if (timestamps.shape != (len(self.streamer.body_joint_names),)
                or timestamps.dtype.kind not in "iu" or np.any(timestamps < 0)
                or np.any(timestamps > np.iinfo(np.int64).max)):
            raise SourceDiscontinuity("SDK joint timestamps must be one nonnegative int64 value per body joint")
        return timestamps.astype(np.int64, copy=True)

    def _read_clock(self) -> tuple[int, np.ndarray | None]:
        """Pin one clock domain; never hop between joints or to the global clock."""
        if self.clock_source == "body_joint":
            timestamps = self._joint_timestamps()
            sequence = int(timestamps[self.clock_joint_index])
            if sequence <= 0:
                raise SourceDiscontinuity("Selected body-joint clock became zero; restart producer after tracking recovers")
            return sequence, timestamps
        aggregate = self.sdk.get_body_timestamp_ns()
        if (isinstance(aggregate, bool) or not isinstance(aggregate, (int, np.integer))
                or not 0 <= aggregate <= np.iinfo(np.int64).max):
            raise SourceDiscontinuity(f"Invalid SDK aggregate body timestamp: {aggregate!r}")
        if self.clock_source == "body" or aggregate > 0:
            self.clock_source = "body"
            return int(aggregate), None
        timestamps = self._joint_timestamps()
        if not np.any(timestamps > 0):
            raise SourceDiscontinuity("Aggregate body timestamp is zero and no positive body-joint timestamp is available")
        # Some clients omit Body.timeStampNs but populate Body.joints[i].t.
        # Pick the newest positive joint once, then keep that exact clock even
        # if another joint later becomes newer. Duplicates/regressions still
        # pass through BodyClock's existing fail-closed validation.
        self.clock_joint_index = int(np.argmax(timestamps))
        self.clock_source = "body_joint"
        name = self.streamer.body_joint_names[self.clock_joint_index]
        print(f"Pico body clock: aggregate timestamp is zero; using joint[{self.clock_joint_index}] {name} "
              "timestamp (nanoseconds, fixed for this producer)", flush=True)
        return int(timestamps[self.clock_joint_index]), timestamps

    def buttons(self) -> tuple[bool, bool]:
        # Independent of body availability, IK, hands, and headset tracking.
        return bool(self.sdk.get_A_button()), bool(self.sdk.get_X_button())

    def diagnostics(self) -> dict:
        """Read-only SDK metadata, never used as a fallback freshness clock."""
        result = {"selected_body_clock": self.clock_source, "selected_joint_index": self.clock_joint_index}
        if self.ignore_sdk_timestamps:
            result.update(selected_body_clock="host_sampling", sdk_timestamps_ignored=True,
                          acquisition_freshness_verified=False,
                          is_body_data_available=bool(self.sdk.is_body_data_available()))
            return result  # Do not read SDK timestamp getters, even for diagnostics.
        if self.clock_joint_index is not None:
            result["selected_joint_name"] = self.streamer.body_joint_names[self.clock_joint_index]
        for name in ("is_body_data_available", "get_body_timestamp_ns", "get_time_stamp_ns",
                     "get_A_button", "get_X_button"):
            getter = getattr(self.sdk, name, None)
            if callable(getter):
                try:
                    value = getter()
                    result[name] = value.item() if isinstance(value, np.generic) else value
                except Exception as error:
                    result[name] = f"{type(error).__name__}: {error}"
        getter = getattr(self.sdk, "get_body_joints_timestamp", None)
        if callable(getter):
            try:
                timestamps = np.asarray(getter())
                result["joint_timestamp_range"] = (
                    [int(timestamps.min()), int(timestamps.max())] if timestamps.size else []
                )
            except Exception as error:
                result["joint_timestamp_range"] = f"{type(error).__name__}: {error}"
        return result

    def read_body(self) -> tuple[int, dict] | None:
        if not self.sdk.is_body_data_available():
            return None
        if self.ignore_sdk_timestamps:
            host_sample = time.monotonic()
            if host_sample - self._last_host_sample < CONTROL_DT:
                return None
            before = int(host_sample * 1_000_000_000)
            self.clock_source = "host_sampling"
        else:
            before, joint_timestamps_before = self._read_clock()
        poses = np.array(self.sdk.get_body_joints_pose(), dtype=np.float64, copy=True)
        if not self.ignore_sdk_timestamps:
            after, joint_timestamps_after = self._read_clock()
            if before != after or (joint_timestamps_before is not None
                                   and not np.array_equal(joint_timestamps_before, joint_timestamps_after)):
                return None  # SDK arrays and body timestamp were a torn snapshot.
        names = self.streamer.body_joint_names
        if poses.shape != (len(names), 7) or not np.isfinite(poses).all():
            raise SourceDiscontinuity("SDK body poses must be finite [body_count, 7]")
        quaternion_norms = np.linalg.norm(poses[:, 3:7], axis=1)
        if not np.allclose(quaternion_norms, 1, atol=1e-3, rtol=0):
            raise SourceDiscontinuity("SDK body pose quaternions must be normalized")
        if np.any(np.abs(poses[:, :3]) > 100):
            raise SourceDiscontinuity("SDK body positions exceed plausible meter bounds")
        body = {
            name: [poses[index, :3].tolist(), poses[index, [6, 3, 4, 5]].tolist()]
            for index, name in enumerate(names)
        }
        if self.ignore_sdk_timestamps:
            self._last_host_sample = host_sample
        return before, self.streamer.coordinate_transform_unity_data(body)

    def close(self) -> None:
        self.sdk.close()


class LatestBodyReader:
    """Bounded single-sample mailbox; pose equality never means clock staleness."""

    def __init__(self, source: XRoboSource, *, max_gap: float = 0.15, poll_interval: float = 0.002):
        self.source = source
        self.clock = BodyClock(max_gap)
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: BodySample | None = None
        self._generation = 0
        self._error: str | None = None
        self._controls: deque[str] = deque(maxlen=32)
        self._previous_buttons = (False, False)
        self._host_anchor: tuple[float, float] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read_loop, name="pico-body-reader", daemon=True)
        self._thread.start()

    def snapshot(self) -> tuple[BodySample | None, int, str | None]:
        with self._lock:
            return self._latest, self._generation, self._error

    def take_controls(self) -> list[str]:
        with self._lock:
            controls = list(self._controls)
            self._controls.clear()
            return controls

    def has_controls(self) -> bool:
        with self._lock:
            return bool(self._controls)

    def poll_controls(self) -> None:
        if not hasattr(self.source, "buttons"):
            return
        buttons = self.source.buttons()
        with self._lock:
            if buttons[1] and not self._previous_buttons[1]:
                self._controls.append("exit")
            elif buttons[0] and not self._previous_buttons[0]:
                self._controls.append("toggle")
            self._previous_buttons = buttons

    def poll_once(self) -> None:
        try:
            raw = self.source.read_body()
            if raw is None:
                return
            sequence, body = raw
            monotonic_time = time.monotonic()
            if getattr(self.source, "ignore_sdk_timestamps", False):
                if self._host_anchor is None:
                    self._host_anchor = (monotonic_time, time.time())
                source_time = self._host_anchor[1] + monotonic_time - self._host_anchor[0]
            else:
                source_time = self.clock.map(sequence, unix_time=time.time(), monotonic_time=monotonic_time)
            if source_time is None:
                return
            sample = BodySample(int(sequence), source_time, monotonic_time, body)
            with self._lock:
                self._latest = sample
                self._error = None
        except Exception as error:
            self.clock.reset()
            with self._lock:
                self._latest = None
                message = str(error)
                if self._error != message:
                    self._generation += 1
                self._error = message

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_controls()
            except Exception:
                with self._lock:
                    self._controls.append("exit")
            self.poll_once()
            self._stop.wait(self.poll_interval)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        # SDK calls are normally non-blocking. Closing SDK also releases a
        # blocked read; join again before returning ownership to the caller.
        self.source.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise RuntimeError("XRobo body reader did not stop after SDK.close()")


class GMRRetargeter:
    """Validate named limited hinges, never assume GMR qpos joint ordering."""

    def __init__(self, retargeter, mujoco, joint_names: list[str]):
        self.retargeter = retargeter
        model = retargeter.model
        self.model = model
        free_joints = np.flatnonzero(np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE))
        if len(free_joints) != 1:
            raise ValueError("GMR model requires exactly one free root")
        self.root_address = int(model.jnt_qposadr[free_joints[0]])
        addresses, limits = [], []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"GMR model missing contract joint {name}")
            if model.jnt_type[joint_id] != int(mujoco.mjtJoint.mjJNT_HINGE) or not model.jnt_limited[joint_id]:
                raise ValueError(f"GMR joint {name} must be a limited hinge")
            bounds = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
            if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"GMR joint {name} has invalid radian bounds")
            addresses.append(int(model.jnt_qposadr[joint_id]))
            limits.append(bounds)
        if len(addresses) != 29 or len(set(addresses)) != 29:
            raise ValueError("GMR mapping requires 29 distinct contract hinges")
        self.addresses = np.asarray(addresses)
        self.limits = np.asarray(limits)
        self.joint_names = tuple(joint_names)
        self.simulator_limits: np.ndarray | None = None

    def configure_simulator_limits(self, parameters: dict) -> None:
        """Install compiled simulator limits by name, in contract joint order."""
        names = parameters["joint_names"]
        lower = np.asarray(parameters["joint_lower"], dtype=np.float64)
        upper = np.asarray(parameters["joint_upper"], dtype=np.float64)
        if (len(names) != 29 or len(set(names)) != 29 or set(names) != set(self.joint_names)
                or lower.shape != (29,) or upper.shape != (29,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(lower >= upper)):
            raise ValueError("Simulator limits require the 29 named contract hinges and finite radian ranges")
        bounds_by_name = dict(zip(names, zip(lower, upper)))
        limits = np.asarray([bounds_by_name[name] for name in self.joint_names])
        if np.any(np.maximum(limits[:, 0], self.limits[:, 0]) >= np.minimum(limits[:, 1], self.limits[:, 1])):
            raise ValueError("Simulator and GMR joint limits have no common valid interval")
        self.simulator_limits = limits
        differences = int(np.count_nonzero(np.any(np.abs(limits - self.limits) > 1e-6, axis=1)))
        print(f"Validated 29 named simulator hinges; {differences} ranges differ from GMR; "
              "every outgoing pose must satisfy both (no clipping)", flush=True)

    def retarget(self, body: dict) -> np.ndarray:
        qpos = np.asarray(self.retargeter.retarget(body, offset_to_ground=True), dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("GMR returned invalid/nonfinite qpos")
        root = qpos[self.root_address:self.root_address + 7].copy()
        if np.any(np.abs(root[:3]) > 100) or not np.isclose(np.linalg.norm(root[3:7]), 1, atol=1e-4, rtol=0):
            raise ValueError("GMR root position/quaternion invalid")
        joints = qpos[self.addresses]
        if np.any(joints < self.limits[:, 0] - 1e-4) or np.any(joints > self.limits[:, 1] + 1e-4):
            raise ValueError("GMR joint positions exceed named hinge radian ranges")
        if self.simulator_limits is not None:
            outside_limits = ((joints < self.simulator_limits[:, 0] - 1e-4)
                              | (joints > self.simulator_limits[:, 1] + 1e-4))
            if np.any(outside_limits):
                invalid_names = [name for name, invalid in zip(self.joint_names, outside_limits) if invalid]
                raise ValueError("GMR pose exceeds simulator MJCF joint limits: " + ", ".join(invalid_names))
        return np.concatenate((root, joints))


def load_adapters(wbc_root: Path, human_height: float) -> tuple[XRoboSource, GMRRetargeter]:
    """Only --run reaches here; do not add WBC's colliding deployment package."""
    paths = external_paths(wbc_root)
    data_processing = str(paths["data_processing"])
    if data_processing not in sys.path:
        sys.path.insert(0, data_processing)
    existing = sys.modules.get("general_motion_retargeting")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve().parent != paths["xrobot_source"].parent:
            raise RuntimeError("A different general_motion_retargeting checkout is already imported")
    sdk = importlib.import_module("xrobotoolkit_sdk")
    missing = [name for name in SDK_GETTERS if not callable(getattr(sdk, name, None))]
    if missing:
        raise RuntimeError("XRoboToolkit SDK missing required API: " + ", ".join(missing))
    xrobot_module = importlib.import_module("general_motion_retargeting.xrobot_utils")
    gmr_module = importlib.import_module("general_motion_retargeting.motion_retarget")
    parameters = importlib.import_module("general_motion_retargeting.params")
    mujoco = importlib.import_module("mujoco")
    from .contract import G1_JOINT_NAMES

    for name, actual in (("model", parameters.ROBOT_XML_DICT["unitree_g1"]),
                         ("config", parameters.IK_CONFIG_DICT["xrobot"]["unitree_g1"])):
        if Path(actual).resolve() != paths[name].resolve():
            raise RuntimeError(f"Unexpected GMR {name} path: {actual}")
    print(json.dumps({"gmr_provenance": _provenance(paths), "human_height_m": human_height,
                      "offset_to_ground": True, "grounding_offset_m": 0.1}, indent=2), flush=True)
    retargeter = gmr_module.GeneralMotionRetargeting(
        src_human="xrobot", tgt_robot="unitree_g1", actual_human_height=human_height,
        solver="daqp", verbose=False,
    )
    mapped = GMRRetargeter(retargeter, mujoco, G1_JOINT_NAMES)
    try:
        streamer = xrobot_module.XRobotStreamer()
        return XRoboSource(sdk, streamer), mapped
    except BaseException:
        sdk.close()
        raise


class TeleopSession:
    """Pure, testable safety state machine around ACK-start/latest-frame publish."""

    def __init__(self, publisher, buffer, retargeter, packet_type, controller_id: str, *, stale_timeout: float = 0.5):
        self.publisher = publisher
        self.buffer = buffer
        self.retargeter = retargeter
        self.packet_type = packet_type
        self.controller_id = controller_id
        self.stale_timeout = stale_timeout
        self.armed = False
        self.previous_a = False
        self.session_id: str | None = None
        self.last_packet = None
        self.last_sequence: int | None = None
        self.generation = 0
        self.armed_at = 0.0
        self.last_publish = -math.inf
        self.last_reference_end = -math.inf
        self.cancel_requested = lambda: False
        self.operator_cancelled = False

    def stop(self, reason: str) -> None:
        if self.last_packet is not None:
            packet = replace(self.last_packet, state="stop", sequence=self.last_packet.sequence + 1,
                             sent_at=time.time(), reference=None)
            try:
                self.publisher.publish(packet)
            finally:
                self.last_packet = None
                self.session_id = None
                self.armed = False
                self.buffer.clear()
        else:
            self.session_id = None
            self.armed = False
            self.buffer.clear()
        print(f"Pico stopped: {reason}; press a new Right A edge to arm", flush=True)

    def step(self, *, right_a: bool, left_x: bool, sample: BodySample | None,
             generation: int, status: dict | None, now: float) -> bool:
        rising_a = right_a and not self.previous_a
        self.previous_a = right_a
        if left_x:
            self.stop("Left X exit")
            return False
        receiver_state = None if status is None else status.get("state")
        # A stopped/stale receiver retains that status until a new start. Only
        # an explicit A edge may fill a *new* buffer and replace that session.
        explicit_restart = self.session_id is None and (rising_a or self.armed)
        healthy_state = receiver_state in ("waiting", "playing", "buffering", "holding")
        restartable_state = explicit_restart and receiver_state in ("stopped", "stale")
        receiver_safe = (
            status is not None and status.get("controller_id") == self.controller_id
            and status.get("stream_mode") == "live" and not status.get("paused", True)
            and (healthy_state or restartable_state)
        )
        if generation != self.generation:
            self.generation = generation
            self.stop("body source clock discontinuity")
            return True
        if not receiver_safe:
            if rising_a:
                print(f"Right A received but receiver is not ready: state={receiver_state}, "
                      f"paused={None if status is None else status.get('paused')}, "
                      f"mode={None if status is None else status.get('stream_mode')}", flush=True)
            if self.armed:
                self.stop("receiver paused/stopped/stale/unavailable")
            return True
        if self.session_id is not None and status.get("session_id") != self.session_id:
            self.stop("receiver session changed")
            return True
        if rising_a:
            if self.armed:
                self.stop("Right A pause")
                return True
            self.buffer.clear()
            self.armed = True
            self.armed_at = now
            self.last_sequence = None if sample is None else sample.source_sequence
            self.last_reference_end = -math.inf
            self.last_publish = -math.inf
            print("Pico armed: filling fresh delayed reference buffer", flush=True)
        if not self.armed:
            return True
        receipt_time = self.armed_at if sample is None else sample.received_monotonic
        if now - receipt_time > self.stale_timeout:
            self.stop("SDK body timestamp stopped progressing")
            return True
        if sample is None or sample.source_sequence == self.last_sequence:
            return True
        self.last_sequence = sample.source_sequence
        try:
            qpos = self.retargeter.retarget(sample.body)
            if self.cancel_requested():
                self.operator_cancelled = True
                self.stop("operator pressed A/X during retargeting")
                return True
            accepted = self.buffer.add(qpos, source_sequence=sample.source_sequence, source_time=sample.source_time)
            if not accepted:
                self.stop("reference buffer rejected source progression")
                return True
            chunk = self.buffer.chunk()
            if chunk is None or now - self.last_publish < CONTROL_DT:
                return True
            reference_time, reference = chunk
            reference_end = reference_time + (len(reference) - 1) * CONTROL_DT
            if reference_end <= self.last_reference_end:
                return True
            starting = self.session_id is None
            if starting:
                self.session_id = uuid.uuid4().hex
            packet = self.packet_type(
                controller_id=self.controller_id, session_id=self.session_id,
                sequence=0 if starting else self.last_packet.sequence + 1,
                sent_at=time.time(), state="start" if starting else "frame", name="pico-gmr",
                source_sequence=self.buffer.source_sequence, source_time=self.buffer.source_time,
                reference_time=reference_time, reference=reference,
            )
            # Keep the pending start for interruption cleanup even before ACK.
            self.last_packet = packet
            if starting:
                self.publisher.start(packet)
            else:
                self.publisher.publish(packet)
            self.last_publish = now
            self.last_reference_end = reference_end
        except ValueError as error:
            self.stop(str(error))
        return True

    def close(self) -> None:
        packet = None if self.last_packet is None else replace(
            self.last_packet, state="stop", sequence=self.last_packet.sequence + 1,
            sent_at=time.time(), reference=None,
        )
        self.last_packet = None
        self.armed = False
        self.buffer.clear()
        self.publisher.close(packet)


def run(args) -> None:
    from .contract import robot_parameters
    from .live_reference import LivePublisher, LiveReferencePacket
    from .teleop_reference import BufferedReference

    client = connect_redis(args.redis_host, args.redis_port, args.redis_db)
    publisher = LivePublisher(client, args.channel, timeout=args.timeout)
    source = None
    reader = None
    session = None
    try:
        simulator_parameters = robot_parameters(args.mjcf)
        source, retargeter = load_adapters(args.wbc_root, args.human_height)
        if getattr(args, "ignore_sdk_timestamps", False):
            source.ignore_sdk_timestamps = True
            print("WARNING: SDK timestamps DISABLED (simulation debugging only). Host sampling at <=50 Hz; "
                  "cached/frozen SDK poses can appear fresh while body_available remains true.", flush=True)
        retargeter.configure_simulator_limits(simulator_parameters)
        # SDK/GMR initialization can exceed the three-second Redis lease.
        # Acquire ownership only once all expensive local setup has completed.
        publisher.acquire()
        status = publisher.wait_ready()
        if status.get("stream_mode") != "live":
            raise RuntimeError("Receiver must be explicitly started in live stream mode")
        reader = LatestBodyReader(source, max_gap=args.max_gap)
        buffer = BufferedReference(status["goal_frames"], filter_tau=args.filter_tau,
                                   jitter_margin=args.jitter_margin, max_gap=args.max_gap,
                                   history_seconds=args.history_seconds, recenter=not args.no_recenter)
        session = TeleopSession(publisher, buffer, retargeter, LiveReferencePacket,
                               status["controller_id"], stale_timeout=args.stale_timeout)
        session.cancel_requested = reader.has_controls
        publisher.cancel_requested = reader.has_controls
        reader.start()
        print("Right A: arm/pause; Left X: exit. No session starts automatically.", flush=True)
        last_diagnostic = -math.inf
        while True:
            sample, generation, error = reader.snapshot()
            # Polling through the publisher also renews its lease while idle.
            status = publisher.read_status()
            if error and generation != session.generation:
                print(f"Pico source: {error}", flush=True)
            controls = reader.take_controls()
            if controls:
                print(f"Pico controller edges: {', '.join(controls)}", flush=True)
            now = time.monotonic()
            if args.diag_interval > 0 and now - last_diagnostic >= args.diag_interval:
                print("PICO DIAG " + json.dumps({
                    "sdk": source.diagnostics(), "source_error": error,
                    "accepted_body_sequence": None if sample is None else sample.source_sequence,
                    "body_receipt_age_s": None if sample is None else round(now - sample.received_monotonic, 3),
                    "armed": session.armed, "receiver_state": None if status is None else status.get("state"),
                    "receiver_reason": None if status is None else status.get("reason"),
                    "receiver_paused": None if status is None else status.get("paused"),
                    "buffer": buffer.diagnostics,
                }), flush=True)
                last_diagnostic = now
            if "exit" in controls:
                session.step(right_a=False, left_x=True, sample=sample, generation=generation,
                             status=status, now=time.monotonic())
                break
            if session.operator_cancelled:
                # The edge already stopped an in-flight IK/ACK transaction.
                # Do not accidentally reinterpret that pause as a fresh arm.
                controls = []
                session.operator_cancelled = False
            for control in controls:
                if control == "toggle":
                    session.previous_a = False
                    session.step(right_a=True, left_x=False, sample=sample, generation=generation,
                                 status=status, now=time.monotonic())
            if not session.step(right_a=False, left_x=False, sample=sample,
                                generation=generation, status=status, now=time.monotonic()):
                break
            time.sleep(0.005)
    finally:
        try:
            if session is not None:
                session.close()
            else:
                publisher.close()
        finally:
            try:
                if reader is not None:
                    reader.close()
                elif source is not None:
                    source.close()
            finally:
                client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wbc-root", type=Path, required=True, help="Existing external WBC checkout; never modified")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF,
                        help="Actual simulator MJCF used to validate named joint limits before publication")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Print offline dependency/provenance diagnostics")
    mode.add_argument("--run", action="store_true", help="Initialize existing SDK and publish to a live local receiver")
    parser.add_argument("--human-height", type=_positive, default=1.6)
    parser.add_argument("--stale-timeout", type=_positive, default=0.5)
    parser.add_argument("--max-gap", type=_positive, default=0.15)
    parser.add_argument("--filter-tau", type=_nonnegative, default=0.03)
    parser.add_argument("--jitter-margin", type=_nonnegative, default=0.04)
    parser.add_argument("--history-seconds", type=_positive, default=1.5)
    parser.add_argument("--no-recenter", action="store_true")
    parser.add_argument("--timeout", type=_positive, default=10.0)
    parser.add_argument("--diag-interval", type=_nonnegative, default=2.0,
                        help="Seconds between read-only SDK/state diagnostics; 0 disables periodic output")
    parser.add_argument("--ignore-sdk-timestamps", action="store_true",
                        help="SIMULATION DEBUG ONLY: sample by host time, skip ALL SDK timestamp reads/checks; "
                             "cannot detect cached/frozen body data")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    args = parser.parse_args(argv)
    try:
        if args.check:
            print(json.dumps(check_environment(args.wbc_root, args.mjcf), indent=2))
        else:
            run(args)
        return 0
    except KeyboardInterrupt:
        print("Pico interrupted; owned session and SDK closed", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, TimeoutError) as error:
        print(f"Pico: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
