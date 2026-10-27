"""Run an exported ONNX policy in standalone MuJoCo, with optional viewer/video.

The policy and motion are independent inputs. No DDS participant, robot channel,
Isaac Lab runtime, or reference-only action replay is used by this program.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import time

import numpy as np

from deployment.common.contract import (
    ACTION_DIM, CONTROL_DT, GOAL_FRAME_DIM, OBSERVATION_DIM,
    build_goals, build_observation, initialize_history, load_reference,
    robot_parameters, sha256_file, update_history,
)
from deployment.simulation.dds_bridge import PHYSICS_DT, SimulationBundle, build_model, reset_from_reference


REPOSITORY = Path(__file__).resolve().parents[2]


class OnnxPolicy:
    """Strict fixed-batch SuperMimic CPU inference, including frozen normalization."""

    def __init__(self, path: Path):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.path = path.resolve(strict=True)
        self.session = onnxruntime.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        inputs = {tensor.name: tensor for tensor in self.session.get_inputs()}
        outputs = self.session.get_outputs()
        if set(inputs) != {"obs_hist", "goal"} or len(outputs) != 1 or outputs[0].name != "action":
            raise ValueError("Expected ONNX inputs obs_hist, goal and one output action")
        history_shape = inputs["obs_hist"].shape
        goal_shape = inputs["goal"].shape
        if (len(history_shape) != 3 or history_shape[0] != 1 or history_shape[2] != OBSERVATION_DIM
                or type(history_shape[1]) is not int or history_shape[1] < 1):
            raise ValueError("obs_hist must have a fixed shape [1, H, 99] with H > 0")
        if (len(goal_shape) != 2 or goal_shape[0] != 1 or type(goal_shape[1]) is not int
                or goal_shape[1] < GOAL_FRAME_DIM or goal_shape[1] % GOAL_FRAME_DIM):
            raise ValueError("goal must have a fixed shape [1, 73*K] with K > 0")
        if outputs[0].shape != [1, ACTION_DIM]:
            raise ValueError("action must have shape [1, 29]")
        if any(tensor.type != "tensor(float)" for tensor in [*inputs.values(), *outputs]):
            raise ValueError("All policy inputs and outputs must be float32 tensors")
        self.history_length = history_shape[1]
        self.goal_frames = goal_shape[1] // GOAL_FRAME_DIM

    def infer(self, history: np.ndarray, goal: np.ndarray) -> np.ndarray:
        action = self.session.run(["action"], {
            "obs_hist": np.ascontiguousarray(history[None], dtype=np.float32),
            "goal": np.ascontiguousarray(goal[None], dtype=np.float32),
        })[0][0]
        if not np.isfinite(action).all():
            raise ValueError("ONNX policy returned a nonfinite action; stopping physics")
        return action


@dataclass
class MotionClip:
    path: Path | None
    reference: np.ndarray
    goals: np.ndarray
    hold_final_velocity: bool = False
    stream_name: str = "waiting"

    @property
    def name(self) -> str:
        return self.path.stem if self.path else self.stream_name

    @classmethod
    def load(cls, path: Path, goal_frames: int, source_fps: float, hold_final_velocity: bool = False) -> MotionClip:
        reference = load_reference(path, src_fps=source_fps)
        return cls(path.resolve(), reference, build_goals(reference, goal_frames, hold_final_velocity=hold_final_velocity),
                   hold_final_velocity)


class PolicySimulation:
    """Reference-clocked 50 Hz controller with effort-limited PD at 500 Hz."""

    def __init__(self, policy: OnnxPolicy, mjcf: Path, clip: MotionClip | None = None):
        import mujoco

        self.policy = policy
        self.mjcf = mjcf.resolve()
        parameters = robot_parameters(mjcf)
        if clip is None:
            reference = np.zeros((2, 71), dtype=np.float32)
            reference[:, 2] = 0.8
            reference[:, 3] = 1.0
            reference[:, 7:36] = parameters["default_joint_position"]
            clip = MotionClip(None, reference, build_goals(reference, policy.goal_frames))
        # Reuse the bridge's named mapping, physics construction, and full-state
        # reset without initializing its transport or requiring a motion bundle.
        self.physics = SimulationBundle(
            directory=mjcf.parent, reference=clip.reference,
            joint_names=tuple(parameters.pop("joint_names")),
            **{name: np.asarray(values, dtype=np.float64) for name, values in parameters.items()},
        )
        self.model, self.data, self.mapping = build_model(mjcf, self.physics, preserve_visual_only=True)
        self.model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
        self.model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
        ground_id = self.model.geom("supermimic_bridge_ground").id
        self.model.geom_rgba[ground_id] = [0.65, 0.68, 0.72, 1]
        self.ghost_data = mujoco.MjData(self.model)
        self.policy_sha256 = sha256_file(policy.path)
        self.mjcf_sha256 = sha256_file(self.mjcf)
        self.substeps = round(CONTROL_DT / self.model.opt.timestep)
        if not math.isclose(self.substeps * self.model.opt.timestep, CONTROL_DT, abs_tol=1e-12):
            raise ValueError("Physics timestep must divide the 20 ms control interval")
        self.reset(clip)

    def reset(self, clip: MotionClip | None = None) -> None:
        if clip is not None:
            self.clip = clip
        self.motion_sha256 = sha256_file(self.clip.path) if self.clip.path else None
        reset_from_reference(self.model, self.data, self.mapping, self.clip.reference[0])
        self.frame = 0
        self.history = None
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.target = self.physics.default_joint_position.copy()
        self.torque_clips = 0
        self.sample_count = 0
        self.joint_error_squared = 0.0
        self.root_error_squared = 0.0
        self.minimum_root_height = float(self.clip.reference[0, 2])
        self.inference_seconds: list[float] = []

    def accept_reference(self, reference: np.ndarray, name: str, *, reset: bool) -> None:
        """Replace external targets, never teleport physics except session start.

        Incoming windows already contain the producer's explicit velocity/tail
        convention. Preserve those values when building the first goal row.
        """
        self.clip = MotionClip(None, reference, build_goals(
            reference, self.policy.goal_frames, hold_final_velocity=True,
        ), stream_name=name)
        self.frame = 0
        if reset:
            self.reset()

    @property
    def finished(self) -> bool:
        return self.frame >= len(self.clip.reference) - 1

    def advance(self) -> None:
        import mujoco

        if self.finished:
            raise RuntimeError("Clip ended; reset or select another clip before stepping")
        mapping = self.mapping
        body_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
            mapping.root_body_id, body_velocity, 1,
        )
        observation = build_observation(
            self.data.qpos[mapping.qpos_addresses], self.data.qvel[mapping.dof_addresses],
            self.data.xquat[mapping.root_body_id], body_velocity[:3], self.previous_action,
            self.clip.reference[self.frame, 3:7], self.physics.default_joint_position,
        )
        self.history = (initialize_history(observation, self.policy.history_length) if self.history is None
                        else update_history(self.history, observation))
        inference_start = time.perf_counter()
        action = self.policy.infer(self.history, self.clip.goals[self.frame])
        self.inference_seconds.append(time.perf_counter() - inference_start)
        # No [-1,1] clipping or extra observation scales: these change the learned
        # policy. Only physical motor torque is bounded, as in the training model.
        self.target = self.physics.default_joint_position + self.physics.action_scale * action
        self.previous_action = action.copy()
        for _ in range(self.substeps):
            torque = (self.physics.kp * (self.target - self.data.qpos[mapping.qpos_addresses])
                      - self.physics.kd * self.data.qvel[mapping.dof_addresses])
            self.torque_clips += int(np.count_nonzero(np.abs(torque) > self.physics.effort_limit))
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[mapping.dof_addresses] = np.clip(
                torque, -self.physics.effort_limit, self.physics.effort_limit,
            )
            mujoco.mj_step(self.model, self.data)
            if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                    or self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
                raise RuntimeError("MuJoCo reported unstable/nonfinite dynamics")
        # mj_step leaves derived body transforms behind the final qpos; make
        # observation, ghost, recording and error measurements agree on the time.
        mujoco.mj_forward(self.model, self.data)
        self.frame += 1
        reference = self.clip.reference[self.frame]
        joint_error = self.data.qpos[mapping.qpos_addresses] - reference[7:36]
        root_error = self.data.qpos[mapping.root_qpos_address:mapping.root_qpos_address + 3] - reference[:3]
        self.sample_count += 1
        self.joint_error_squared += float(joint_error @ joint_error)
        self.root_error_squared += float(root_error @ root_error)
        self.minimum_root_height = min(self.minimum_root_height, float(self.data.xpos[mapping.root_body_id, 2]))

    def update_ghost(self, anchor_xy: bool) -> None:
        import mujoco

        self.ghost_data.qpos[:] = self.data.qpos
        root_address = self.mapping.root_qpos_address
        self.ghost_data.qpos[root_address:root_address + 7] = self.clip.reference[self.frame, :7]
        self.ghost_data.qpos[self.mapping.qpos_addresses] = self.clip.reference[self.frame, 7:36]
        if anchor_xy:
            self.ghost_data.qpos[root_address:root_address + 2] = self.data.qpos[root_address:root_address + 2]
        mujoco.mj_forward(self.model, self.ghost_data)

    def add_ghost(self, scene, options) -> None:
        import mujoco

        starting_index = scene.ngeom
        mujoco.mjv_addGeoms(
            self.model, self.ghost_data, options, mujoco.MjvPerturb(),
            mujoco.mjtCatBit.mjCAT_DYNAMIC, scene,
        )
        for geometry in scene.geoms[starting_index:scene.ngeom]:
            geometry.rgba[:] = [0.3, 0.65, 1.0, 0.3]
            geometry.matid = -1
            geometry.transparent = True

    def metrics(self) -> dict:
        return {
            "motion": str(self.clip.path) if self.clip.path else None,
            "reference_name": self.clip.name, "policy": str(self.policy.path),
            "policy_sha256": self.policy_sha256, "motion_sha256": self.motion_sha256,
            "mjcf": str(self.mjcf), "mjcf_sha256": self.mjcf_sha256,
            "control_dt": CONTROL_DT, "physics_dt": PHYSICS_DT,
            "physics": "MuJoCo explicit effort-clipped PD; self-collision disabled",
            "control_steps": self.sample_count, "simulation_seconds": float(self.data.time),
            "reference_exhausted": self.finished,
            "hold_final_velocity": self.clip.hold_final_velocity,
            "joint_position_rmse": math.sqrt(self.joint_error_squared / max(1, self.sample_count * ACTION_DIM)),
            "root_position_rmse": math.sqrt(self.root_error_squared / max(1, self.sample_count * 3)),
            "minimum_root_height": self.minimum_root_height, "torque_clips": self.torque_clips,
            "inference_p95_ms": float(np.percentile(self.inference_seconds, 95) * 1000) if self.inference_seconds else 0.0,
            "success_evaluated": False,
        }


class TrajectoryRecorder:
    """Control-rate actual/reference states; clip resets have explicit segment IDs."""

    def __init__(self):
        self.samples: list[tuple] = []

    def append(self, simulation: PolicySimulation, segment: int, total_steps: int, reference_sequence: int = -1,
               *, source_sequence: int = -1, reference_time: float = -1, source_time: float = -1,
               holding: bool = False) -> None:
        mapping = simulation.mapping
        root_address = mapping.root_qpos_address
        self.samples.append((
            segment, total_steps * CONTROL_DT, simulation.clip.name, simulation.frame,
            simulation.data.qpos[root_address:root_address + 7].copy(),
            simulation.data.qpos[mapping.qpos_addresses].copy(),
            simulation.clip.reference[simulation.frame].copy(), simulation.previous_action.copy(),
            reference_sequence,
            source_sequence, reference_time, source_time, holding,
        ))

    def save(self, path: Path) -> None:
        columns = list(zip(*self.samples))
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{
                name: np.asarray(values) for name, values in zip(
                    ("segment", "time", "clip", "frame", "root_pose_wxyz", "joint_position", "reference", "action",
                     "reference_sequence", "source_sequence", "reference_time", "source_time", "holding"),
                    columns,
                )
            })


class VideoRecorder:
    """Fixed simulation-time FPS, independent of slow inference or GUI pacing."""

    def __init__(self, simulation: PolicySimulation, path: Path, fps: int):
        import mujoco

        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Video recording requires ffmpeg on PATH")
        simulation.model.vis.global_.offwidth = 960
        simulation.model.vis.global_.offheight = 540
        self.renderer = mujoco.Renderer(simulation.model, height=540, width=960)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.5, 135, -15
        self.options = mujoco.MjvOption()
        self.process = subprocess.Popen([
            executable, "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "960x540", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path),
        ], stdin=subprocess.PIPE)

    def capture(self, simulation: PolicySimulation, show_ghost: bool) -> None:
        self.camera.lookat[:] = simulation.data.xpos[simulation.mapping.root_body_id]
        self.renderer.update_scene(simulation.data, camera=self.camera, scene_option=self.options)
        if show_ghost:
            simulation.add_ghost(self.renderer.scene, self.options)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self) -> None:
        try:
            self.process.stdin.close()
            if self.process.wait(timeout=30) != 0:
                raise RuntimeError("ffmpeg failed to write the simulation video")
        except BaseException:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            raise
        finally:
            self.renderer.close()


def validate_output(path: Path | None) -> None:
    if path is not None and (os.path.lexists(path) or not path.parent.is_dir()):
        raise ValueError(f"Output must not exist and its parent must exist: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    motions = parser.add_mutually_exclusive_group()
    motions.add_argument("--motion", type=Path, help="A single LAFAN1 G1 CSV")
    motions.add_argument("--motion-dir", type=Path, help="CSV library; [ / ] switch clips and reset")
    motions.add_argument("--listen", action="store_true", help="Load policy first, then accept Redis reference windows")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--channel", default="supermimic:g1")
    parser.add_argument("--reference-timeout", type=float, default=0.5,
                        help="Fresh-reference timeout in seconds; freeze simulation and require a new session")
    parser.add_argument("--stream-mode", choices=["csv", "live"], default="csv",
                        help="csv ACKs one step per packet; live consumes buffered references on independent ticks")
    parser.add_argument("--hold-timeout", type=float, default=0.12,
                        help="Maximum live lookahead underrun hold in seconds before freezing")
    parser.add_argument("--max-reference-delay", type=float, default=1.0)
    parser.add_argument("--max-control-lateness", type=float, default=0.15,
                        help="Freeze live simulation if one tick/render operation exceeds this wall-time budget")
    parser.add_argument("--clip", help="Initial clip stem when using --motion-dir")
    parser.add_argument("--mjcf", type=Path, default=REPOSITORY / "data/g1_description/g1_29dof.xml")
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--hold-final-velocity", action="store_true",
                        help="Legacy clip-tail goals for checkpoints trained before the stationary-tail fix (Job 74284)")
    parser.add_argument("--headless", action="store_true", help="No window; run until clip end or --seconds")
    parser.add_argument("--seconds", type=float, help="Total simulated seconds across resets/clips")
    parser.add_argument("--loop", action="store_true", help="Explicitly reset full state at clip end")
    parser.add_argument("--paused", action="store_true", help="Start viewer paused")
    parser.add_argument("--no-ghost", action="store_true")
    parser.add_argument("--global-reference", action="store_true", help="Do not anchor reference XY to actual robot")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace physics to wall time")
    parser.add_argument("--record", type=Path, help="Save actual ONNX closed-loop states to NPZ")
    parser.add_argument("--metrics", type=Path, help="Save per-segment diagnostic JSON")
    parser.add_argument("--video", type=Path, help="Save MP4 (ffmpeg required); also works headless")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS, must divide 50")
    arguments = parser.parse_args(argv)
    if arguments.seconds is not None and (not math.isfinite(arguments.seconds) or arguments.seconds < CONTROL_DT):
        parser.error("--seconds must be finite and at least 0.02")
    if arguments.headless and (arguments.paused or (arguments.loop and arguments.seconds is None)):
        parser.error("headless runs cannot start paused; --loop requires --seconds")
    if arguments.fps < 1 or 50 % arguments.fps:
        parser.error("--fps must be a positive divisor of 50")
    if arguments.clip and not arguments.motion_dir:
        parser.error("--clip requires --motion-dir")
    streaming = arguments.listen or not (arguments.motion or arguments.motion_dir)
    if streaming and (arguments.loop or arguments.no_realtime or arguments.hold_final_velocity):
        parser.error("streaming uses real-time pacing; loop/tail conventions belong to the sender")
    if not math.isfinite(arguments.reference_timeout) or arguments.reference_timeout <= 0:
        parser.error("--reference-timeout must be finite and positive")
    for setting in (arguments.hold_timeout, arguments.max_reference_delay, arguments.max_control_lateness):
        if not math.isfinite(setting) or setting <= 0:
            parser.error("live delay/hold/timing budgets must be finite and positive")
    if arguments.stream_mode == "live" and not streaming:
        parser.error("--stream-mode live cannot be combined with direct-file playback")
    try:
        for output in (arguments.record, arguments.metrics, arguments.video):
            validate_output(output)
        outputs = [path.resolve() for path in (arguments.record, arguments.metrics, arguments.video) if path]
        if len(set(outputs)) != len(outputs):
            raise ValueError("Recording, metrics and video must use different output paths")
        clip_paths = ([arguments.motion] if arguments.motion else sorted(arguments.motion_dir.glob("*.csv"))
                      if arguments.motion_dir else [])
        if not clip_paths and not streaming:
            raise ValueError("No motion CSVs found")
        clip_index = 0
        if arguments.clip:
            clip_index = next((index for index, path in enumerate(clip_paths) if path.stem == arguments.clip), -1)
            if clip_index < 0:
                raise ValueError(f"Clip not found: {arguments.clip}")
        import torch

        torch.set_num_threads(1)
        policy = OnnxPolicy(arguments.policy)
        clip = (MotionClip.load(clip_paths[clip_index], policy.goal_frames, arguments.src_fps, arguments.hold_final_velocity)
                if not streaming else None)
        simulation = PolicySimulation(policy, arguments.mjcf, clip)
        print(f"ONNX CPU | H={policy.history_length} K={policy.goal_frames} | {simulation.clip.name}", flush=True)
        if streaming:
            print("Space: pause | R: stop reference session | G: ghost | F: follow | Esc: close", flush=True)
        else:
            print("Space: pause | R: reset | [ / ]: previous/next | G: ghost | F: follow | Esc: close", flush=True)
        print("Physics: explicit limited PD, 2 ms; policy: 20 ms; no DDS or hardware connections", flush=True)
        if streaming:
            from deployment.streaming.receiver import run_stream_simulation

            run_stream_simulation(simulation, arguments)
        else:
            run_simulation(simulation, clip_paths, clip_index, arguments)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f"sim2sim failed: {error}\n")
    return 0


def run_simulation(simulation: PolicySimulation, clip_paths: list[Path], clip_index: int, arguments) -> None:
    import mujoco

    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    recording = TrajectoryRecorder() if arguments.record else None
    reports = []
    segment = total_steps = 0
    paused, show_ghost, follow = arguments.paused, not arguments.no_ghost, True
    step_limit = round(arguments.seconds / CONTROL_DT) if arguments.seconds is not None else None
    frame_stride = 50 // arguments.fps
    deadline = time.perf_counter()
    last_video_step = -1
    with ExitStack() as resources:
        viewer = None
        if not arguments.headless:
            if os.name == "posix" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise RuntimeError("No display available; use --headless (optionally --video with MUJOCO_GL=egl)")
            import mujoco.viewer

            viewer = resources.enter_context(mujoco.viewer.launch_passive(
                simulation.model, simulation.data, key_callback=key_events.put,
                show_left_ui=False, show_right_ui=False,
            ))
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 135, -15
        video = VideoRecorder(simulation, arguments.video, arguments.fps) if arguments.video else None
        if video:
            resources.callback(video.close)
        if recording:
            recording.append(simulation, segment, total_steps)
        try:
            while viewer is None or viewer.is_running():
                while not key_events.empty():
                    key = key_events.get()
                    if key == 32:
                        paused = not paused
                    elif key in (ord("G"), ord("g")):
                        show_ghost = not show_ghost
                    elif key in (ord("F"), ord("f")):
                        follow = not follow
                    elif key in (ord("R"), ord("r"), ord("["), ord("]")):
                        reports.append(simulation.metrics())
                        if key in (ord("["), ord("]")):
                            clip_index = (clip_index + (1 if key == ord("]") else -1)) % len(clip_paths)
                        clip = MotionClip.load(clip_paths[clip_index], simulation.policy.goal_frames,
                                               arguments.src_fps, arguments.hold_final_velocity)
                        simulation.reset(clip)
                        segment += 1
                        paused = False
                        if recording:
                            recording.append(simulation, segment, total_steps)
                        print(f"Reset: {clip.name}", flush=True)
                if step_limit is not None and total_steps >= step_limit:
                    break
                if simulation.finished and arguments.loop:
                    reports.append(simulation.metrics())
                    simulation.reset()
                    segment += 1
                    if recording:
                        recording.append(simulation, segment, total_steps)
                simulation.update_ghost(anchor_xy=not arguments.global_reference)
                if viewer:
                    with viewer.lock():
                        if follow:
                            viewer.cam.lookat[:] = simulation.data.xpos[simulation.mapping.root_body_id]
                        viewer.user_scn.ngeom = 0
                        if show_ghost:
                            simulation.add_ghost(viewer.user_scn, viewer.opt)
                    if hasattr(viewer, "set_texts"):
                        viewer.set_texts((int(mujoco.mjtFontScale.mjFONTSCALE_100), int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                                          f"{simulation.clip.name}  {simulation.frame * CONTROL_DT:.2f}s"
                                          + ("  END (R or [ / ])" if simulation.finished else "  PAUSED" if paused else ""),
                                          "Space pause | R reset | [ ] clips | G ghost | F follow"))
                    viewer.sync()
                if (video and total_steps % frame_stride == 0 and total_steps != last_video_step
                        and not paused and not simulation.finished):
                    video.capture(simulation, show_ghost)
                    last_video_step = total_steps
                if simulation.finished and not arguments.loop:
                    if viewer is None:
                        break
                    paused = True
                if paused:
                    time.sleep(CONTROL_DT)
                    deadline = time.perf_counter()
                    continue
                simulation.advance()
                total_steps += 1
                if recording:
                    recording.append(simulation, segment, total_steps)
                if total_steps % 50 == 0:
                    print(f"{simulation.clip.name} t={simulation.data.time:.2f}s "
                          f"joint_rmse={simulation.metrics()['joint_position_rmse']:.4f}rad", flush=True)
                if not arguments.no_realtime:
                    deadline += CONTROL_DT
                    time.sleep(max(0, deadline - time.perf_counter()))
                    # Slow inference never skips physics/reference frames and never
                    # produces a long catch-up burst after a pause or window drag.
                    deadline = max(deadline, time.perf_counter() - CONTROL_DT)
        except KeyboardInterrupt:
            pass
        finally:
            reports.append(simulation.metrics())
            report = {"segments": reports, "total_control_steps": total_steps, "success_evaluated": False}
            if recording:
                recording.save(arguments.record)
            if arguments.metrics:
                with arguments.metrics.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
            print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
