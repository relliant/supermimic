"""Opt-in native SDK2/real MuJoCo transport checks, never hardware validation.

Run serially with SUPERMIMIC_DDS_TEST=1 and SUPERMIMIC_NATIVE_BINARY pointing
to the built supermimic_g1 executable. The current Python must provide torch,
ONNX, MuJoCo and SDK2/CycloneDDS. No dependency is imported before opt-in.
Only loopback domains 42-45 are opened. The checkpoint and spherical model
are synthetic, not a learned policy, authentic G1 geometry or sim2real evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest


if os.environ.get("SUPERMIMIC_DDS_TEST") != "1":
    pytest.skip("native DDS integration requires SUPERMIMIC_DDS_TEST=1", allow_module_level=True)


REPOSITORY = Path(__file__).resolve().parents[2]
BRIDGE_READY = "NATIVE_TEST_BRIDGE_READY"
# Synchronize *after* MJCF compilation and both DDS endpoint initializations.
# This leaves the actual CLI, state writer, fault clocks and physics unchanged.
BRIDGE_DRIVER = """
import select
import sys
from deployment import sim_bridge

original_run = sim_bridge.SimulationBridge.run

def synchronized_run(self, seconds):
    print("NATIVE_TEST_BRIDGE_READY", flush=True)
    readable, _, _ = select.select([sys.stdin], [], [], 20)
    if not readable or sys.stdin.readline().strip() != "RUN":
        raise RuntimeError("native test startup handshake timed out")
    return original_run(self, seconds)

sim_bridge.SimulationBridge.run = synchronized_run
sim_bridge.main()
"""


@dataclass(frozen=True)
class SyntheticFixture:
    binary: Path
    bundle: Path
    mjcf: Path
    safety_profile: Path
    manifest: dict


@pytest.fixture(scope="module")
def synthetic_directory(tmp_path_factory):
    parent = tmp_path_factory.mktemp("synthetic_native_dds")
    # Remove the relatively large checkpoint/ONNX files even on fixture failure.
    with tempfile.TemporaryDirectory(prefix="artifacts_", dir=parent) as directory:
        yield Path(directory)


@pytest.fixture(scope="module")
def synthetic_fixture(synthetic_directory):
    if os.environ.get("PYTEST_XDIST_WORKER", "master") != "master":
        pytest.fail("native DDS tests must run serially without pytest-xdist")
    if "CYCLONEDDS_URI" in os.environ:
        pytest.fail("unset CYCLONEDDS_URI before isolated DDS integration tests")
    binary_setting = os.environ.get("SUPERMIMIC_NATIVE_BINARY")
    if not binary_setting:
        pytest.fail("opt-in also requires SUPERMIMIC_NATIVE_BINARY")
    binary = Path(binary_setting).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        pytest.fail(f"native executable is missing or not executable: {binary}")

    # A shared advisory lock also rejects concurrent independent pytest runs.
    lock_path = Path(tempfile.gettempdir()) / "supermimic-native-dds-integration.lock"
    with lock_path.open("a") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pytest.fail("another native DDS integration suite is already running")

        import numpy as np
        import torch
        from deployment import contract
        from deployment.export_policy import Policy, export_bundle

        directory = synthetic_directory
        checkpoint = directory / "SYNTHETIC_ZERO_ACTION_NOT_LEARNED.pt"
        motion = directory / "SYNTHETIC_REST.csv"
        mjcf = directory / "SYNTHETIC_SPHERE_NOT_G1.xml"
        previous_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(20261026)
                policy = Policy(history_length=10, goal_frames=10).eval()
                policy.dec[-1].weight.zero_()
                policy.dec[-1].bias.zero_()
                for normalizer in (policy.obs_norm, policy.goal_norm):
                    normalizer.mean.zero_()
                    normalizer.var.fill_(1)
                    normalizer.count.fill_(64)
                torch.save({"pi": policy.state_dict(), "args": {
                    "hist": 10, "goal_frames": 10, "amp": "none",
                    "synthetic_untrained": True,
                }}, checkpoint)

            # All child centers coincide with their hinges: no gravity torques.
            # The free root sphere rests on the bridge's z=0 ground plane.
            children = "".join(
                f'<body name="synthetic_child_{index}">'
                f'<joint name="{name}" type="hinge" limited="true" range="-3 3"/>'
                '<geom type="sphere" size="0.005" mass="0.0001"/></body>'
                for index, name in enumerate(contract.G1_JOINT_NAMES)
            )
            mjcf.write_text(
                '<mujoco model="SYNTHETIC_NOT_G1_GEOMETRY"><compiler angle="radian"/>'
                '<worldbody><body name="synthetic_pelvis" pos="0 0 0.3">'
                '<freejoint name="synthetic_root"/>'
                '<geom type="sphere" size="0.3" mass="10"/>'
                f'{children}</body></worldbody></mujoco>', encoding="ascii",
            )
            rows = np.zeros((51, 36), dtype=np.float32)
            rows[:, 2] = 0.3
            rows[:, 6] = 1  # Input CSV uses xyzw, unlike reference.bin.
            rows[:, 7:36] = contract.DEFAULT_JOINT_POSITION
            np.savetxt(motion, rows, delimiter=",")
            bundle = directory / "bundle"
            export_bundle(checkpoint, motion, bundle, mjcf, src_fps=50, parity_samples=2)
        finally:
            torch.set_num_threads(previous_thread_count)

        manifest = json.loads((bundle / "manifest.json").read_text())
        assert manifest["history_length"] == manifest["goal_frames"] == 10
        assert manifest["frame_count"] == 51
        reference = np.fromfile(bundle / "reference.bin", dtype="<f4").reshape(51, 71)
        np.testing.assert_array_equal(reference[:, 36:], 0)
        parity = json.loads((bundle / "parity.json").read_text())
        for sample in parity["samples"]:
            np.testing.assert_array_equal(sample["action"], np.zeros(29))

        profile = {
            "schema_version": 1, "hardware_validated": False,
            "manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
            "pelvis_imu_verified": False, "safe_exit_verified": False,
            "imu_frame": "pelvis", "sensor_to_pelvis_wxyz": [1, 0, 0, 0],
            "position_lower": manifest["joint_lower"], "position_upper": manifest["joint_upper"],
            "velocity_limit": manifest["velocity_limit"], "torque_limit": manifest["effort_limit"],
            "kp": manifest["kp"], "kd": manifest["kd"],
            "target_rate_limit": [3] * 29, "damping_kd": [2] * 29,
            "state_timeout_ms": 100, "command_timeout_ms": 100, "inference_timeout_ms": 20,
            "maximum_temperature": 70, "maximum_tilt_error": 0.6,
            "preparation_seconds": 5, "shutdown_seconds": 0.2,
            "maximum_start_joint_speed": 0.2, "maximum_reference_start_speed": 0.2,
            "maximum_start_position_error": 0.1,
        }
        safety_profile = directory / "UNVALIDATED_SIMULATION_ONLY.json"
        safety_profile.write_text(json.dumps(profile), encoding="ascii")
        yield SyntheticFixture(binary, bundle, mjcf, safety_profile, manifest)


@contextmanager
def managed_process(arguments: list[str], log_path: Path, *, input_pipe: bool = False):
    """Keep output bounded by the test duration and reap children on every path."""
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            arguments, cwd=REPOSITORY, env=environment,
            stdin=subprocess.PIPE if input_pipe else subprocess.DEVNULL,
            stdout=log_stream, stderr=subprocess.STDOUT, text=True,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process.stdin is not None:
                process.stdin.close()


def wait_for_log(process: subprocess.Popen, log_path: Path, marker: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = log_path.read_text(encoding="utf-8")
        if marker in output:
            return
        if process.poll() is not None:
            pytest.fail(f"process exited before {marker!r}:\n{output}")
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {marker!r}:\n{log_path.read_text()}")


def run_isolated_pair(fixture: SyntheticFixture, directory: Path, *, domain: int,
                      monitor: bool = False, drop_state: bool = False) -> tuple[int, str, dict, float]:
    assert domain in (42, 43, 44, 45)
    metrics_path = directory / "bridge_metrics.json"
    bridge_log = directory / "bridge.log"
    controller_log = directory / "controller.log"
    bridge_arguments = [
        sys.executable, "-u", "-c", BRIDGE_DRIVER,
        "--bundle", str(fixture.bundle), "--mjcf", str(fixture.mjcf),
        "--interface", "lo", "--domain", str(domain), "--freeze-before-command",
        "--seconds", "2", "--metrics", str(metrics_path),
    ]
    if drop_state:
        bridge_arguments.extend(["--drop-state-after", "0.3"])
    controller_arguments = [
        str(fixture.binary), "--bundle", str(fixture.bundle),
        "--interface", "lo", "--domain", str(domain),
        "--mode", "monitor" if monitor else "simulation", "--seconds",
        "1" if monitor else ("0.8" if drop_state else "0.4"),
    ]
    if not monitor:
        controller_arguments.extend([
            "--safety-profile", str(fixture.safety_profile), "--simulation-autostart",
        ])
    with managed_process(bridge_arguments, bridge_log, input_pipe=True) as bridge:
        wait_for_log(bridge, bridge_log, BRIDGE_READY)
        with managed_process(controller_arguments, controller_log) as controller:
            # MODE is emitted after native DDS initialization, before waiting for
            # state. Thus fault thresholds cannot expire during bundle/ORT load.
            wait_for_log(controller, controller_log, "MODE ")
            bridge_released = time.monotonic()
            bridge.stdin.write("RUN\n")
            bridge.stdin.flush()
            try:
                return_code = controller.wait(timeout=15)
                controller_finished = time.monotonic()
                bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pytest.fail(f"native integration timed out\nCONTROLLER:\n{controller_log.read_text()}"
                            f"\nBRIDGE:\n{bridge_log.read_text()}")
        assert bridge.returncode == 0, bridge_log.read_text()
    output = controller_log.read_text()
    assert metrics_path.is_file(), bridge_log.read_text()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["published_states"] > 0, metrics
    assert metrics["rejected_commands"] == 0, metrics
    assert metrics["success_evaluated"] is False
    # A conservative lower bound on live bridge time after the controller exit.
    # Process teardown time is NOT evidence of watchdog/physics execution.
    remaining_seconds = metrics["wall_seconds"] - (controller_finished - bridge_released)
    return return_code, output, metrics, remaining_seconds


def test_native_monitor_never_publishes_commands(synthetic_fixture, tmp_path):
    return_code, output, metrics, _ = run_isolated_pair(
        synthetic_fixture, tmp_path, domain=42, monitor=True,
    )
    assert return_code == 0, output
    assert "MONITOR" in output, output
    assert metrics["accepted_commands"] == 0, metrics
    assert metrics["frozen_steps"] > 0 and metrics["physics_steps"] == 0, metrics


def test_native_simulation_runs_zero_action_policy(synthetic_fixture, tmp_path):
    return_code, output, metrics, _ = run_isolated_pair(synthetic_fixture, tmp_path, domain=43)
    # Deadline faults on shared hosts are real failures, not retries or skips.
    assert return_code == 0, output
    assert "TRACKING" in output and "STOPPED completed" in output, output
    assert metrics["accepted_commands"] > 0 and metrics["physics_steps"] > 0, metrics
    assert metrics["maximum_applied_torque"] <= max(synthetic_fixture.manifest["effort_limit"])


def test_native_state_loss_trips_controller_watchdog(synthetic_fixture, tmp_path):
    return_code, output, metrics, _ = run_isolated_pair(
        synthetic_fixture, tmp_path, domain=44, drop_state=True,
    )
    assert return_code != 0, output
    assert "TRACKING" in output and "State watchdog expired" in output, output
    assert metrics["accepted_commands"] > 0, metrics
    assert metrics["stale_steps"] > 0, metrics


def test_native_command_loss_after_exit_uses_bridge_watchdog(synthetic_fixture, tmp_path):
    return_code, output, metrics, remaining_seconds = run_isolated_pair(
        synthetic_fixture, tmp_path, domain=45,
    )
    assert return_code == 0, output
    assert remaining_seconds > 0.1, "bridge must outlive native shutdown and its 100 ms watchdog"
    assert metrics["accepted_commands"] > 0 and metrics["stale_steps"] > 0, metrics
    assert metrics["physics_steps"] > 0, "command loss must not refreeze the simulator"
    assert metrics["simulation_seconds"] > 0.4, metrics


def test_native_teardown_while_reader_callbacks_are_active(synthetic_fixture, tmp_path):
    # Regression for CycloneDDS 0.10.2's prevent_callbacks assertion: dropping
    # the last reader reference during a callback can destroy it on that same
    # callback thread. The native controller must explicitly close the reader
    # on its owner thread while the simulator is still publishing live states.
    for repetition in range(5):
        directory = tmp_path / f"close_{repetition}"
        directory.mkdir()
        return_code, output, metrics, remaining_seconds = run_isolated_pair(
            synthetic_fixture, directory, domain=43,
        )
        assert return_code == 0, f"teardown repetition {repetition}:\n{output}"
        assert "WRITER_EXIT" in output and metrics["published_states"] > 0
        assert remaining_seconds > 0.1, "state publisher must remain active past controller teardown"


def test_native_unvalidated_hardware_profile_refused_before_dds(synthetic_fixture, tmp_path):
    # This deliberately nonexistent interface is never queried: profile loading
    # precedes ChannelFactory::Init. Do not replace it with a hardware interface.
    log_path = tmp_path / "hardware_refusal.log"
    arguments = [
        str(synthetic_fixture.binary), "--bundle", str(synthetic_fixture.bundle),
        "--mode", "hardware", "--enable-motors", "--interface", "example-do-not-use",
        "--domain", "0", "--safety-profile", str(synthetic_fixture.safety_profile), "--seconds", "0.4",
    ]
    with managed_process(arguments, log_path) as controller:
        try:
            return_code = controller.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail(f"unvalidated hardware preflight did not refuse promptly:\n{log_path.read_text()}")
    output = log_path.read_text()
    assert return_code != 0, output
    assert "Hardware requires a reviewed profile, verified pelvis IMU and verified safe exit" in output, output
    assert "MODE " not in output, output
