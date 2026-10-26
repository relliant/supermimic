# G1 EDU 29-DoF onboard deployment

This is an independent C++17 / ONNX Runtime CPU / SDK2 controller for an **x86_64
onboard computer**. It does not use Isaac Lab, CUDA, TensorRT, a world model, a
critic, SONIC's encoder, or a cloud inference service at runtime.

**Status: development implementation, not hardware-qualified.** Source-level
conformance and synthetic tests are not evidence that a trained policy can
balance a real G1. No robot was connected or actuated while implementing this.
See [VALIDATION.md](VALIDATION.md) for the actual checks and remaining gates.

## 1. Export a specific checkpoint and reference clip

Run from this repository's root, using an environment with
`deployment/requirements.txt` installed. The exporter itself is Isaac-free; it
does not import `supermimic.robot` or instantiate a simulator. Its inference-only
policy definition is checked against the original training source in tests.

```bash
python -m deployment.export_policy \
  --checkpoint /path/to/ckpt_014000.pt \
  --motion /path/to/lafan1_g1/SELECTED_VALIDATED_CLIP.csv \
  --mjcf /path/to/g1_description/g1_29dof.xml \
  --output /existing/parent/g1-policy-bundle \
  --parity-samples 12
```

The checkpoint must contain `pi` and `args.hist` / `args.goal_frames`, with
initialized normalizers. Only trusted checkpoint provenance should be used;
loading uses `weights_only=True`, never an unsafe-pickle fallback. Output must
not already exist. One bundle contains:

- `policy.onnx`: float32 deterministic policy, embedded frozen normalization.
- `reference.bin`: little-endian float32 `[frames, 71]`.
- `goals.bin`: little-endian float32 `[frames, 73 * goal_frames]`.
- `parity.json`: nontrivial input/state/history/action conformance samples.
- `manifest.json`: schema, SHA256 hashes, source/checkpoint/CSV identities, joint
  order, limits, trained actuator constants and timing.

The 71-value reference row is **not** `MDP.flatten_state`: it is root position
3, root quaternion **wxyz** 4, joint position 29, joint velocity 29, world linear
velocity 3, world angular velocity 3. No full dataset copy or USD is needed onboard.
Keep each bundle immutable and share it between runs; vary run configuration,
not code-directory copies. Preserve LAFAN1's license restrictions.

Training may use bf16, but export and CPU inference use float32. Before hardware
use compare the selected learned checkpoint's original evaluation and exported
closed-loop behavior; a synthetic FP32 parity pass does not address that shift.

## 2. Build the native runtime

Install a compiler, CMake, Eigen3, nlohmann-json and OpenSSL development packages.
On Ubuntu the package names are `build-essential cmake libeigen3-dev
nlohmann-json3-dev libssl-dev`. Obtain a matching **CPU ONNX Runtime C/C++ SDK**
for the onboard x86_64 machine, containing `include/` and `lib/`. Record its
version. No dependencies are fetched automatically by CMake.

```bash
cmake -S deployment -B .cache/deployment/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_ROOT=/path/to/onnxruntime-linux-x64
cmake --build .cache/deployment/build -j
ctest --test-dir .cache/deployment/build --output-on-failure
```

For SDK2, additionally provide the existing SDK2 source/install prefix:

```bash
cmake -S deployment -B .cache/deployment/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_ROOT=/path/to/onnxruntime-linux-x64 \
  -DSUPERMIMIC_WITH_SDK2=ON \
  -DSDK2_ROOT="$HOME/Project/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2"
cmake --build .cache/deployment/build -j
```

The external reference tree is not changed or copied. SDK2's architecture-matched
library and CycloneDDS headers/libraries must be present. All binaries must be
matched to the actual onboard ABI and libraries; a desktop build is not an
onboard latency qualification. See [THIRD_PARTY.md](THIRD_PARTY.md).

## 3. Verify numerical equivalence and inference timing

```bash
.cache/deployment/build/supermimic_offline \
  --bundle /path/to/g1-policy-bundle \
  --parity /path/to/g1-policy-bundle/parity.json \
  --benchmark 1000
.cache/deployment/build/test_safety /path/to/g1-policy-bundle
python -m pytest deployment/tests -q
```

The offline program verifies payload hashes and compares observation, goal,
history and action values. It reports p50/p95/p99/max observation-to-inference
latency and counts overruns of **20 ms**. Repeat on the actual onboard computer
under expected load. Do not enable control if its worst-case timing misses the
budget. ORT options are applied before constructing the session; tensors are
preallocated. Core-only tests can build with `-DSUPERMIMIC_WITH_ORT=OFF` without
SDK2 or ONNX Runtime.

Contract details that must not be changed:

- Policy at 50 Hz; default inputs `obs_hist[1,10,99]`, `goal[1,730]`, output 29.
  Actual history/horizon come from the checkpoint.
- History is oldest-first; first observation is repeated, previous action zero.
- Raw joint velocity and gyro are **not** additionally scaled by `OBS_SCALE`.
- Rotation 6D is `R00,R01,R10,R11,R20,R21`, not concatenated complete columns.
- Future goals start at `f+1`, end at `f+K`, and clamp within the selected clip.
- Commands use `q0 + action_scale * action`, with per-joint exported constants.
  Neither `[-1,1]` action clipping nor SONIC's action scale is compatible.
- Limits/slew limiting are explicit safety deviations. The previous-action input
  is updated from the bounded target in the same dimensionless convention.
- Initial yaw alignment is fixed at playback start, not recomputed every frame.

## 4. Isolated DDS + MuJoCo closed loop

See [SIMULATION.md](SIMULATION.md) for the bridge and metrics. It requires MuJoCo
and the official Python SDK2 package with its compatible CycloneDDS library. The
bridge accepts only **interface `lo` and a nonzero DDS domain**; both processes
reject `CYCLONEDDS_URI` overrides. Do not use a robot network interface.

First terminal, from the repository root:

```bash
python -m deployment.sim_bridge \
  --bundle /path/to/g1-policy-bundle \
  --mjcf /path/to/g1_description/g1_29dof.xml \
  --interface lo --domain 42 --seconds 60 --freeze-before-command \
  --metrics /existing/output/bridge.json
```

Second terminal (the **same controller executable** later used onboard):

```bash
.cache/deployment/build/supermimic_g1 \
  --mode simulation --bundle /path/to/g1-policy-bundle \
  --interface lo --domain 42 \
  --safety-profile deployment/config/simulation.json \
  --simulation-autostart
```

The supplied profile is **simulation-only** and fails hardware validation. The
bridge uses explicit effort-clipped PD at 2 ms versus training's implicit PD and
5 ms substeps; these are recorded modeling differences, not equivalence claims.
The original self-collision-disabled setting is preserved for comparison and is
itself a sim-to-real risk. The bridge reports tracking error and whether physics
advanced; it deliberately does not declare learned-policy success automatically.

For opt-in local integration tests, set `SUPERMIMIC_DDS_TEST=1` and
`SUPERMIMIC_NATIVE_BINARY` to the built executable, then run
`deployment/tests/test_native_integration.py` sequentially. No hardware test is
included or enabled by these variables.

## 5. Hardware monitoring and review template

**Default mode is monitor.** It subscribes to `rt/lowstate` only, enforces CRC,
and never creates a command publisher or motion-switcher client. The operator
must choose the correct onboard interface/domain; examples intentionally do not
guess the robot's network configuration:

```bash
.cache/deployment/build/supermimic_g1 \
  --bundle /path/to/g1-policy-bundle \
  --interface YOUR_VERIFIED_INTERFACE --domain YOUR_VERIFIED_DOMAIN

python -m deployment.make_safety_profile \
  --bundle /path/to/g1-policy-bundle --output /existing/output/g1-review.json
```

The generated template is **not** an approved configuration: model-derived
limits/gains, example slew rates and all three validation flags require review.
Hardware profiles must explicitly set every 29-joint array and timing/exit
threshold, remain within model hard limits, and match the exact manifest SHA256.
The template refuses to overwrite an existing review.

Only a calibrated, **rigid pelvis IMU** is implemented. `sensor_to_pelvis_wxyz`
means the rotation mapping sensor-frame vectors into the pelvis frame; both
orientation and gyro are transformed. A torso IMU across the articulated waist
is rejected, not silently treated as pelvis. If the actual firmware exposes only
torso measurements, implement and verify articulated angular-velocity compensation
before proceeding. Joint zero calibration must already match the training model.

## 6. Hardware execution boundary

There is deliberately no unattended hardware-start example. After separate
authorization and supervised validation, the program requires all of:

- Explicit `--mode hardware --enable-motors`, a non-loopback interface, and a
  profile with `hardware_validated`, `pelvis_imu_verified`, `safe_exit_verified`.
- A terminal operator typing **ARM**, fresh valid state and a checked inactive
  stock controller. The program **never releases** the stock controller; use the
  manufacturer's supervised ownership procedure outside this program.
- A near-rest reference and robot, a bounded preparation ramp, then a separate
  **PLAY** confirmation and matching initial joint pose.

Motor publication is 500 Hz; the writer independently checks state/command age,
motor faults, both signs of velocity/torque, temperature, finite values, predicted
PD torque and reference-relative tilt. Tilt supervision also applies during
preparation and the PLAY wait. Rate limiting uses actual target update spacing,
preventing late-loop catch-up bursts. No implicit replay or clip looping occurs.

`x` then Enter, Ctrl-C or a fault stops tracking. The writer attempts bounded
damping **only with fresh finite feedback**, then ceases publication. Stale state
causes it to stop sending instead of synthesizing feedback-derived commands.
This is **not guaranteed to be a safe stand or a recovered stock controller**:
loss-of-command behavior and damping must be verified with protection equipment
and an independent hardware emergency stop. SIGKILL, power loss and a stuck DDS
Write cannot be made safe by a software watchdog in this process alone.

Before any unprotected operation, review motor gains, firmware, load, mounting,
joint limits/zeroes, startup velocities, self-collision, pose transitions, exits
and timing. Do not start with jumping, fighting, falling/get-up or high-dynamic
clips. A pose ramp cannot reproduce reference nonzero initial velocities.

## Artifacts

Keep exported bundles and review profiles outside tracked source, or under
ignored `.cache/` while testing. Save source revision/patch, model/reference
hashes, review configuration, timing and simulation results. Do not retain full
per-experiment copies of the repository, SDKs or training environments. The
implementation does not modify remote training code or any robot files.
