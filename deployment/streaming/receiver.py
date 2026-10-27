"""Local Redis reference-window receiver for the simulation-only ONNX runner.

No target packet contains a path to load or executable content. On missing or
invalid input the simulation freezes; that behavior is NOT a hardware failsafe.
"""

from __future__ import annotations

from contextlib import ExitStack
from collections import deque
import json
import os
import queue
import time
import uuid

from deployment.common.contract import CONTROL_DT
from deployment.streaming.protocol import ExpiredReferenceError, connect_redis, decode, packet_key, status_key
from deployment.streaming.live import LiveReferenceReceiver


class ReferenceReceiver:
    """One fresh, validated packet permits one 20 ms simulation transition."""

    def __init__(self, controller_id: str, goal_frames: int, timeout: float):
        self.controller_id = controller_id
        self.goal_frames = goal_frames
        self.timeout = timeout
        self.session_id = None
        self.sequence = -1
        self.state = "waiting"
        self.reason = "waiting for a new reference session"
        self.last_received = None
        self.horizon_mode = None
        self.closed_sessions: set[str] = set()

    def stop(self, reason: str, state: str = "stopped") -> None:
        if self.session_id:
            self.closed_sessions.add(self.session_id)
        self.state, self.reason = state, reason

    def check_timeout(self, now: float) -> None:
        if self.state == "playing" and now - self.last_received > self.timeout:
            self.stop("reference stream timed out; start a new sender session", "stale")

    def accept(self, payload: bytes | None, simulation, *, wall_time: float, monotonic_time: float) -> bool:
        """Acknowledge after physics, not after merely seeing the Redis key."""
        self.check_timeout(monotonic_time)
        if payload is None:
            return False
        try:
            packet = decode(payload, self.goal_frames, now=wall_time, max_age=self.timeout)
        except ExpiredReferenceError:
            if self.state == "playing":
                self.stop("reference packet expired; start a new sender session", "stale")
            return False
        except ValueError as error:
            self.check_timeout(monotonic_time)
            if self.state not in ("stopped", "stale", "error"):
                self.stop(str(error), "error")
            return False
        if packet.controller_id != self.controller_id:
            self.check_timeout(monotonic_time)
            return False
        if packet.session_id in self.closed_sessions:
            return False
        is_start = packet.state == "start" and packet.session_id != self.session_id
        if not is_start and (packet.session_id != self.session_id or packet.sequence <= self.sequence):
            self.check_timeout(monotonic_time)
            return False
        if packet.state == "stop":
            self.sequence = packet.sequence
            self.stop("sender finished or requested stop")
            return False
        if not is_start and self.state != "playing":
            return False
        if is_start:
            if self.session_id:
                self.closed_sessions.add(self.session_id)
            self.session_id = packet.session_id
            self.sequence = -1
        try:
            simulation.accept_reference(packet.reference, packet.name, reset=is_start)
            simulation.advance()
        except (ValueError, RuntimeError) as error:
            self.stop(str(error), "error")
            return False
        self.sequence = packet.sequence
        self.last_received = monotonic_time
        self.state, self.reason = "playing", ""
        self.horizon_mode = packet.horizon_mode
        return True

    def status(self, *, paused: bool) -> dict:
        return {
            "schema": "supermimic.reference.status", "version": 1,
            "controller_id": self.controller_id, "goal_frames": self.goal_frames,
            "control_dt": CONTROL_DT, "state": self.state, "reason": self.reason,
            "session_id": self.session_id, "sequence": self.sequence,
            "updated_at": time.time(), "paused": paused,
        }


def run_stream_simulation(simulation, arguments) -> None:
    """Keep the policy/viewer alive across independent file or teleop producers."""
    from deployment.simulation.sim2sim import TrajectoryRecorder, VideoRecorder
    import redis

    controller_id = uuid.uuid4().hex
    live = arguments.stream_mode == "live"
    receiver = (LiveReferenceReceiver(
        controller_id, simulation.policy.goal_frames, source_timeout=arguments.reference_timeout,
        hold_timeout=arguments.hold_timeout, max_delay=arguments.max_reference_delay,
    ) if live else ReferenceReceiver(controller_id, simulation.policy.goal_frames, arguments.reference_timeout))
    if live:
        # Initialize lazy ORT/torch/MuJoCo work before advertising a real-time
        # receiver. This discarded simulation-only warm-up is fully reset and
        # is not a received control tick or an executed robot command.
        simulation.advance()
        simulation.reset()
    client = connect_redis(arguments.redis_host, arguments.redis_port, arguments.redis_db)
    controller_lock = arguments.channel + ":controller"
    if not client.set(controller_lock, controller_id, nx=True, ex=3):
        client.close()
        raise RuntimeError("Another controller owns this channel; use a different --channel")
    compare_delete = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
    compare_expire = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('expire', KEYS[1], 3) else return 0 end"
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    reports = []
    recording = TrajectoryRecorder() if arguments.record else None
    segment = total_steps = 0
    step_limit = round(arguments.seconds / CONTROL_DT) if arguments.seconds else None
    paused, follow, show_ghost = arguments.paused, True, not arguments.no_ghost
    was_paused = paused
    last_state = None
    deadline = time.perf_counter()
    last_lock_refresh = 0.0
    last_draw = 0.0
    control_times: deque[float] = deque(maxlen=101)
    max_tick_seconds = 0.0
    print(f"Listening on Redis {arguments.redis_host}:{arguments.redis_port} channel={arguments.channel}", flush=True)
    print("Waiting: launch deployment.teleop_pico" if live else
          "Waiting: launch python -m deployment.send_motion --motion MOTION.csv in another terminal", flush=True)
    try:
        with ExitStack() as resources:
            viewer = None
            if not arguments.headless:
                if os.name == "posix" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                    raise RuntimeError("No display available; use --headless for stream validation")
                import mujoco.viewer

                viewer = resources.enter_context(mujoco.viewer.launch_passive(
                    simulation.model, simulation.data, key_callback=key_events.put,
                    show_left_ui=False, show_right_ui=False,
                ))
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 135, -15
            video = VideoRecorder(simulation, arguments.video, arguments.fps) if arguments.video else None
            if video:
                resources.callback(video.close)
            try:
                while viewer is None or viewer.is_running():
                    while not key_events.empty():
                        key = key_events.get()
                        if key == 32:
                            paused = not paused
                            if live:
                                receiver.stop("viewer pause/resume requires a new Pico session")
                        elif key in (ord("G"), ord("g")):
                            show_ghost = not show_ghost
                        elif key in (ord("F"), ord("f")):
                            follow = not follow
                        elif key in (ord("R"), ord("r")):
                            receiver.stop("reset requested; launch a new sender")
                    now = time.monotonic()
                    if not live and was_paused and not paused and receiver.last_received is not None:
                        # A deliberate viewer pause stops the reference clock as
                        # well; it must not consume the stream watchdog budget.
                        receiver.last_received = now
                    was_paused = paused
                    if now - last_lock_refresh >= 1:
                        if not client.eval(compare_expire, 1, controller_lock, controller_id):
                            raise RuntimeError("Controller channel ownership lost")
                        last_lock_refresh = now
                    previous_session = receiver.session_id
                    previous_report = simulation.metrics() if receiver.session_id else None
                    advanced = False
                    if not paused:
                        if live:
                            receiver.ingest(client.get(packet_key(arguments.channel)), simulation, wall_time=time.time())
                            advanced = receiver.tick(simulation, wall_time=time.time(), monotonic_time=now)
                        else:
                            advanced = receiver.accept(client.get(packet_key(arguments.channel)), simulation,
                                                       wall_time=time.time(), monotonic_time=now)
                    if receiver.session_id != previous_session:
                        if previous_report:
                            reports.append(previous_report)
                        segment += 1
                        control_times.clear()
                    if advanced:
                        control_times.append(now)
                        simulation.update_ghost(anchor_xy=not arguments.global_reference)
                        if video and total_steps % (50 // arguments.fps) == 0:
                            video.capture(simulation, show_ghost)
                        total_steps += 1
                        if recording:
                            recording.append(
                                simulation, segment, total_steps, receiver.sequence,
                                source_sequence=receiver.source_sequence if live else -1,
                                reference_time=receiver.reference_time if live else -1,
                                source_time=receiver.source_time if live else -1,
                                holding=live and receiver.state == "holding",
                            )
                        # Offline CSV playback waits for the ACK while inference
                        # or rendering runs. Watchdog idle time starts after that
                        # work, not at the beginning of an expensive first step.
                        if not live:
                            receiver.last_received = time.monotonic()
                    status = receiver.status(paused=paused)
                    frequency = ((len(control_times) - 1) / max(1e-6, control_times[-1] - control_times[0])
                                 if len(control_times) > 1 else 0.0)
                    status.update(control_frequency_hz=frequency,
                                  max_tick_seconds=max_tick_seconds)
                    client.set(status_key(arguments.channel), json.dumps(status), ex=2)
                    if receiver.state != last_state:
                        print(f"Stream {receiver.state}: {receiver.reason or simulation.clip.name}", flush=True)
                        last_state = receiver.state
                    if viewer and (not live or now - last_draw >= 1 / 25):
                        import mujoco

                        with viewer.lock():
                            if follow:
                                viewer.cam.lookat[:] = simulation.data.xpos[simulation.mapping.root_body_id]
                            viewer.user_scn.ngeom = 0
                            if show_ghost and receiver.session_id:
                                simulation.add_ghost(viewer.user_scn, viewer.opt)
                        if hasattr(viewer, "set_texts"):
                            diagnostic = "Space pause | R stop session | G ghost | F follow"
                            if live:
                                age = status['source_age_seconds']
                                delay = status['reference_delay_seconds']
                                diagnostic += (f"\nsource age={age:.3f}s" if age is not None else "\nsource age=--")
                                diagnostic += (f" delay={delay:.3f}s" if delay is not None else " delay=--")
                                diagnostic += f" buffer={status['buffer_frames']} control={frequency:.1f}Hz"
                            viewer.set_texts((int(mujoco.mjtFontScale.mjFONTSCALE_100), int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                                              f"{receiver.state.upper()} {'PAUSED' if paused else ''}  {simulation.clip.name}",
                                              diagnostic))
                        viewer.sync()
                        last_draw = now
                    if advanced and not live:
                        receiver.last_received = time.monotonic()
                    tick_seconds = time.monotonic() - now
                    max_tick_seconds = max(max_tick_seconds, tick_seconds)
                    if live and advanced and tick_seconds > arguments.max_control_lateness:
                        receiver.stop("live control/render deadline exceeded; rearm without a catch-up burst", "stale")
                    if step_limit is not None and total_steps >= step_limit:
                        break
                    deadline += CONTROL_DT
                    time.sleep(max(0, deadline - time.perf_counter()))
                    deadline = max(deadline, time.perf_counter() - CONTROL_DT)
            except KeyboardInterrupt:
                pass
            except redis.exceptions.RedisError as error:
                receiver.stop(f"Redis disconnected: {type(error).__name__}", "stale")
                print("Reference transport disconnected; simulation stopped. Restart controller and rearm.", flush=True)
            finally:
                report = simulation.metrics()
                report.update(session_id=receiver.session_id, last_sequence=receiver.sequence,
                              horizon_mode="buffered" if live else receiver.horizon_mode, stream_state=receiver.state,
                              stream_mode=arguments.stream_mode, max_tick_seconds=max_tick_seconds)
                if live:
                    report["live_status"] = receiver.status(paused=paused)
                reports.append(report)
                if recording:
                    recording.save(arguments.record)
                if arguments.metrics:
                    with arguments.metrics.open("x", encoding="utf-8") as stream:
                        json.dump({"segments": reports, "total_control_steps": total_steps,
                                   "success_evaluated": False}, stream, indent=2, allow_nan=False)
    finally:
        try:
            if client.get(controller_lock) == controller_id.encode():
                receiver.stop("controller closed")
                client.set(status_key(arguments.channel), json.dumps(receiver.status(paused=paused)), ex=2)
            client.eval(compare_delete, 1, controller_lock, controller_id)
        except redis.exceptions.RedisError:
            # Ownership expires on its own; never mask a recorded transport failure.
            pass
        finally:
            client.close()
