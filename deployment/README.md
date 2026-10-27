# Deployment: export, simulation and Pico teleoperation

Run all commands from the **SuperMimic repository root**. Model export and
reference generation are independent: load ONNX once, then start either a CSV
sender or the Pico/GMR producer. Neither sim2sim workflow connects to a robot.
Native G1 hardware support is separate and **not hardware-qualified**.

## Pico quick start

Prerequisites: the Pico app is connected to an already-running XRoboToolkit PC
Service, the Python SDK is installed, robot assets exist under `data/`, and
`checkpoints/1026_baseline_lafan/ckpt_020000.onnx` has been exported. If using a
Mac SSH forward for Pico connectivity, keep it running. These commands do not
start/restart PC Service or modify network interfaces.

### 1. Check dependencies and Redis

```bash
cd /home/sia_luo/Project/supermimic
export LD_LIBRARY_PATH="$PWD/XRoboToolkit-PC-Service-Pybind/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.venv/bin/python -m deployment.teleop_pico \
  --wbc-root "$HOME/Project/WholeBodyControl-PolicyTraining" --check

redis-cli -h 127.0.0.1 -p 6379 ping
```

Inspect the check report's `available` fields; generating a report is not a
device-readiness test. Redis should return `PONG`. Only if no Redis service is
running, start one in another terminal:

```bash
redis-server --bind 127.0.0.1 --port 6379 --protected-mode yes --save "" --appendonly no
```

Optional producer dependencies (SDK native binding is installed separately):

```bash
uv pip install --python .venv/bin/python -r deployment/requirements-pico.txt
```

### 2. Terminal A: model and MuJoCo viewer

```bash
cd /home/sia_luo/Project/supermimic
.venv/bin/python -m deployment.sim2sim \
  --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --stream-mode live --channel supermimic:pico
```

Wait for `Stream waiting`. No motion file is needed. Close/restart an older
viewer if it was launched without `--stream-mode live`.

### 3. Terminal B: Pico reference producer

The command below includes the **SDK-timestamp bypass used in the operator's
successful local simulation check**. Set `--human-height` to the actual height
in metres, not blindly to the example value.

```bash
cd /home/sia_luo/Project/supermimic
export LD_LIBRARY_PATH="$PWD/XRoboToolkit-PC-Service-Pybind/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.venv/bin/python -m deployment.teleop_pico \
  --wbc-root "$HOME/Project/WholeBodyControl-PolicyTraining" \
  --human-height 1.6 \
  --channel supermimic:pico \
  --ignore-sdk-timestamps \
  --run
```

**Timestamp bypass is simulation debugging only:** poses are sampled with host
time. If the SDK remains body-available while returning a frozen cached pose,
the producer cannot detect that source failure. Omit `--ignore-sdk-timestamps`
to enable the default body/joint timestamp validation. Pose, joint-limit,
transport and control-timeout checks remain enabled in both modes.

- Right controller **A**: arm and fill a new buffer; press again to pause/stop.
- Left controller **X**: stop and exit the producer.
- Viewer **Space / R**: pause or stop the session; resume with a fresh A press.
- Viewer **G / F**: toggle the reference ghost / camera following.

The solid robot is ONNX-controlled physics; blue transparent geometry is the
reference. For K=10, defaults require at least **260 ms of measured samples**
before start (200 ms horizon + jitter/derivative guards); total response latency
also includes acquisition, IK and filtering. A short underrun holds reference
while closed-loop control continues; timeout freezes the simulator and requires
rearming. Do not disable additional protections to mask tracking errors.

See [Pico setup and troubleshooting](docs/PICO.md) for SDK ABI requirements,
body-clock compatibility, grounding assumptions, timing and recording.

## Export an independent ONNX

```bash
.venv/bin/python -m deployment.export_policy \
  --checkpoint checkpoints/1026_baseline_lafan/ckpt_020000.pt \
  --output checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --parity-samples 12
```

No motion or MJCF is required for model-only export. Existing output is refused;
skip this step if that ONNX already exists. For CSV playback and video output,
see [standalone sim2sim](docs/SIM2SIM.md). For native SDK2 deployment bundles,
build instructions and hardware boundaries, see [native runtime](docs/NATIVE.md).

## Code organization

- `common/`: shared observation/goal contracts, robot constants, bundle integrity.
- `export/`: policy loading, ONNX export, parity and bundle generation.
- `simulation/`: standalone ONNX/MuJoCo runner and isolated DDS bridge.
- `streaming/`: CSV/live protocols, Redis publication, receiver scheduling.
- `teleop/`: Pico SDK/GMR adapter and timestamped reference buffering.
- `safety/`: offline generation of explicitly unvalidated hardware profiles.
- `native/`: C++ runtime (`src/`, `include/`) and native tests (`tests/`).
- `config/`: simulation-only safety profile; not a hardware approval.
- `tests/`: Python unit and opt-in integration tests.
- `docs/`: detailed guides, protocol descriptions and historical validation receipts.

The six root Python files are **CLI entry points only**. Existing commands
(`deployment.export_policy`, `sim2sim`, `sim_bridge`, `send_motion`, `teleop_pico`,
`make_safety_profile`) stay stable; import implementation APIs from their
functional subpackages rather than these wrappers. Package imports do not
initialize a device or eagerly start a simulator.

`cmake -S deployment -B .cache/deployment/build` remains supported. Native
executables stay in the build root; no SDKs are fetched by CMake.

## Validation and further reading

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment/tests -q

SUPERMIMIC_ONNX=checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest deployment.tests.test_live_integration -v
```

The opt-in live test uses recorded poses, not a connected Pico. It creates and
cleans its own temporary loopback Redis instance. Normal tests never actuate a
robot. Keep bundles, checkpoints and generated videos outside tracked source.

- [Reference protocols and timing](docs/REFERENCE_STREAM.md)
- [Isolated DDS simulation](docs/SIMULATION.md)
- [Historical native validation](docs/VALIDATION.md)
- [Third-party dependencies](docs/THIRD_PARTY.md)
