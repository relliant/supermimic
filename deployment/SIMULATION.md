# Isolated MuJoCo SDK2 bridge

`sim_bridge.py` is a standalone, headless simulation process. It does not import
Isaac Lab, the training package, the exporter, or the hardware runtime. It does
not execute a policy: a separate controller must consume `rt/lowstate` and write
HG `rt/lowcmd` messages. There is no direct ONNX Runtime transport in this bridge.

## Safety boundary

- Only DDS interface `lo` is accepted. Non-loopback interfaces are rejected before
  SDK2 is imported or any participant is initialized.
- The default domain is **42**. Domain zero is forbidden; accepted domains are
  integers 1 through 232. The controller must use the same nonzero domain.
- Any `CYCLONEDDS_URI` environment override is rejected, including empty or
  apparently loopback-only XML. Includes and competing interface rules make
  accepting arbitrary DDS configuration unsafe. Unset it explicitly before use.
- Run as a separate process, not inside another application's already initialized
  SDK2 channel factory. Do not connect a robot or choose a hardware interface.
- No DDS participant was initialized while implementing or running the tests.

Use an existing Python environment with NumPy, MuJoCo providing `MjSpec`, and the
official `unitree_sdk2py` package plus its CycloneDDS dependencies. MuJoCo 3.11.0
and 3.14.0 were used for the local physics tests. SDK2 is imported lazily by `connect()`;
helper tests and `--help` do not require it. No packages are installed by the
bridge, and no model or mesh files are downloaded or copied into the repository.

## Inputs and invocation

Supply the generated **robot** XML, not a separate scene with its own floor.
`MjSpec.from_file()` loads the original absolute path so nested XML includes and
compiler-relative meshes keep their original resolution bases. The bridge adds
a ground plane at world z=0; existing scene floors are not removed.

Example for an explicitly requested, isolated local run:

```bash
python deployment/sim_bridge.py \
  --bundle /absolute/path/to/exported_bundle \
  --mjcf /absolute/path/to/g1_29dof.xml \
  --interface lo --domain 42 \
  --freeze-before-command --seconds 10 \
  --metrics /existing/output/directory/simulation.json
```

Start the separate controller in its simulation-only mode with the same
loopback interface/domain. Never relax the bridge's isolation gate to accommodate
a hardware-default controller. Metrics are always printed to stdout; `--metrics`
also writes JSON to an existing parent directory. Startup warnings go to stderr.

The bundle must have a schema-version-1 `manifest.json` containing `control_dt`
equal to 0.02, positive `frame_count`, 29 `joint_names` in fixed G1 SDK order, and finite
29-element `default_joint_position`, `kp`, `kd`, `effort_limit`, `velocity_limit`,
`action_scale`, `joint_lower`, and `joint_upper` arrays. Gains, effort/velocity
limits and action scales must be strictly positive. Lower bounds must be strictly
less than upper bounds. As in `deployment.contract.verify_bundle`,
`history_length` and `goal_frames` are positive integers, `observation_dim=99`,
`action_dim=29`, and `goal_dim=73*goal_frames`. Filenames must be the fixed
`policy.onnx`, `reference.bin`, and `goals.bin` names.

SHA256 validation is mandatory, not an optional simulation bypass. The manifest's
`sha256` object must contain exactly `policy.onnx`, `reference.bin`, `goals.bin`,
and `parity.json`; every payload must be a nonsymlink regular file with matching
SHA256. Both binary files must have the declared dimensions and finite float32
values. This validation is implemented locally to avoid importing the contract
module's torch dependency. It does not authenticate the manifest itself or
execute/validate the ONNX graph or parity samples.

`reference.bin` must be exactly `frame_count * 71 * 4` bytes of little-endian
float32 rows, with columns ordered as follows:

1. Root position in world coordinates: 3 values.
2. Root quaternion, normalized **wxyz**: 4 values.
3. Joint positions in manifest order: 29 values.
4. Joint velocities in manifest order: 29 values.
5. Root linear velocity in world coordinates: 3 values.
6. Root angular velocity in world coordinates: 3 values.

The bridge uses the manifest and reference for simulation, but requires the
complete package for integrity validation. The separate policy controller uses
`goals.bin`; this bridge checks its declared dimensions and finite values but
neither generates goals nor interprets their features.
Joint addresses and the root freejoint are looked up, never inferred from fixed
qpos/dof offsets. Exactly one freejoint is required; each named motor must be a
scalar joint underneath it.

## Physics and command semantics

- Gravity is `[0, 0, -9.81]`. Physics and lowstate publication are **500 Hz**,
  `dt=0.002`; policy control remains 50 Hz, `control_dt=0.02`. This deliberately
  differs from the original reference simulation's `dt=0.005`.
- Robot self-collision is disabled: all robot geoms use `contype=1` and
  `conaffinity=2`; the ground/environment uses 2 and 1. Masks are set before
  compilation. Explicit robot-robot contact pairs are rejected because they
  bypass those masks. All robot geoms, including originally visual-only geoms,
  receive these masks; choose the supplied robot XML accordingly.
  The specification ancestry walk stops explicitly at `worldbody`: MuJoCo 3.11
  does not expose a safe null parent above it. Traversing beyond that boundary
  corrupted specification state and caused a reproducible compilation segfault.
- Per-joint armature is `kp / (20*pi)^2`. Other passive damping, friction, limits,
  inertias and contact settings remain those of the supplied XML. This is not a
  claim of dynamics equivalence with Isaac Lab or real hardware.
- All native XML actuators are removed. Every physics step explicitly applies
  `kp*(q_target-q) + kd*(dq_target-dq) + tau_feedforward` through
  `qfrc_applied`, with the final torque clipped to `+-effort_limit`.
  This is a deliberate departure from the original position-only bridge.
- HG commands must use `mode_pr=0` (PR). Motor mode 1 enables control; mode 0
  disables that motor's PD and feedforward; other motor modes are rejected.
- Nonfinite targets, torques or gains, and negative gains, reject the entire
  message and immediately clear the old mailbox. Targets are capped to joint
  position and velocity bounds; nonnegative gains are capped to manifest `kp`
  and `kd`; feedforward torque is capped before the final effort clamp as well.
- The default command watchdog is 0.1 wall-clock seconds, configurable with
  `--command-timeout`. Missing, invalid or stale commands produce effort-clipped
  `-manifest.kd*dq` damping, never continued old position tracking/feedforward.
  Physical velocity itself is not artificially clipped.

State publication uses HG lowstate with PR mode, configurable `--mode-machine`
(default 5), named joint q/dq/ddq, and actual commanded torque as `tau_est`.
Unused motor entries and temperatures are zeroed. Quaternion is root **wxyz**.
Gyro is the root's body-axis angular velocity obtained with
`mj_objectVelocity(..., mjOBJ_XBODY, ..., local=1)`: MuJoCo returns angular then
linear components, and `mjOBJ_BODY` would instead use inertial principal axes.
The uncalibrated accelerometer is explicitly zero, not a physical IMU model.
Lowstate CRC is populated using the SDK2 CRC helper. Tick advances by 2 ms per
writer iteration, including while the starting pose is frozen.

## Reset, timing and faults

Initialization resets the full state to **reference frame 0**, including root
pose, named joint q/dq, root world linear velocity, and world angular velocity
converted into the local freejoint rotational qvel convention. Simulation time
starts at zero; it is not initialized from an external controller phase.

`--freeze-before-command` holds that full initial state until the first valid
command, publishing live ticks without advancing physics/reference time. After
the first valid command, command loss invokes damping; it does not refreeze.
Without this flag, the robot immediately simulates under watchdog damping.

`--seconds` is elapsed **wall time**, including any frozen interval. A monotonic
clock schedules fixed 2 ms physics steps; physics dt is never adapted or skipped
to hide lateness. The loop can catch up when late, but exits at the wall deadline
and reports maximum scheduling lateness. Python/DDS timing is not hard real time.

Fault-injection options are elapsed wall-time thresholds from loop start:

- `--drop-state-after 2`: stop lowstate publication after two seconds. The
  controller must independently detect stale/missing state and stop; this bridge
  cannot prove that behavior by merely continuing to run.
- `--drop-command-after 2`: ignore received commands after two seconds. The
  bridge's command watchdog must then switch to damping.

Use each fault in a separate, explicitly authorized loopback test. A bridge-only
unit test does not certify the separate controller's watchdog.

## Metrics and verification scope

JSON includes accepted/rejected/dropped commands, published states, stale/frozen
steps, actual torque clipping counts and maxima, scheduling lateness, command
interarrival distributions, and receipt-to-physics-step command age. The latter
is **not** end-to-end controller/transport latency: HG commands do not carry a
matching controller timestamp in this implementation.

Root XYZ/orientation and 29 joint-position RMSE are measured against the reference
frame selected by simulation time. The final frame is held after the clip ends,
with `reference_exhausted` reported; reference playback does not silently loop.
Final actual joint/root state and applied torques are included. Metrics always
report `success_evaluated=false`: no-crash is not a tracking, FK-parity, or
closed-loop acceptance criterion. There is no cross-engine link-FK comparison
without a matching external FK oracle and robot assets.

Run focused tests without DDS:

```bash
python -m pytest deployment/tests/test_sim_bridge.py -q
python deployment/sim_bridge.py --help
```

The tests create temporary 29-joint XML/mesh/include fixtures at runtime; no robot
model is committed. They cover scrambled named-joint mapping with a nonzero root
offset, frame-0 reset, world/body angular-velocity conversion, collision masks,
native-actuator removal, torque/gain/target caps, invalid command rejection,
stale damping, loopback/domain/environment gates, increasing frozen ticks,
measured offline physics errors, and HG field population using test messages.
They also cover every payload's SHA256 and symlink rejection, package schema
mismatches, invalid hashed goals, and 100 sequential model builds with explicit
garbage collection to catch specification-lifetime/traversal regressions.
They do not initialize DDS or substitute for an SDK2 wire test, a real G1 asset
run, an ONNX policy closed loop, or a hardware test.

After the MuJoCo 3.11 traversal fix, plugin-autoload-disabled verification using
`/home/sia_luo/miniconda3/envs/haic/bin/python` passed **62 bridge tests** and the
combined `deployment/tests` suite passed **81 tests**, with three existing
torch ONNX deprecation warnings. The 62 bridge tests also passed on MuJoCo
3.14.0. No DDS initialization or robot connection was performed for these checks.
