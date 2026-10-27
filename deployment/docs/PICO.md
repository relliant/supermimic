# Pico buffered teleoperation: local MuJoCo only

The ONNX controller and Pico producer are independent processes. Pico body
tracking is converted by the existing WholeBodyControl GMR checkout into full
robot states; the controller consumes delayed, already-observed future targets.
This does not use SONIC tokens or the other repository's 35D Redis observations.
No robot-facing SDK2 channel, motor command, or physical robot is involved.

## Quickstart: confirmed simulation-debug path

Run all commands from the repository root. Prerequisites: an exported ONNX
policy, a compatible SDK/GMR Python environment, local Redis (`redis-cli ping`
returns `PONG`), and the Pico application connected to a **manually started**
XRoboToolkit PC Service. Keep the working Mac forwarding setup, when used, and
SDK shared-library environment (including `LD_LIBRARY_PATH`) intact. Do not
start a duplicate Redis or PC Service, or clear a shared Redis database.

Terminal 1: start the local controller/viewer:

```bash
.venv/bin/python -m deployment.sim2sim \
  --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --stream-mode live --channel supermimic:pico
```

Terminal 2: use the interpreter/environment with the working SDK and GMR:

```bash
.venv/bin/python -m deployment.teleop_pico \
  --wbc-root "$HOME/Project/WholeBodyControl-PolicyTraining" \
  --human-height 1.6 --channel supermimic:pico \
  --ignore-sdk-timestamps --run
```

`1.6` is a placeholder in metres: replace it with the operator's actual height.
Wait for `WAITING`, then press right **A** to arm; A again pauses/stops, and left
**X** exits the producer. The operator confirmed this timestamp-bypass runtime
path after installing the SDK; it is **not** full live-tracking qualification.
The optional debug flag disables SDK acquisition-freshness verification:
`sdk_timestamps_ignored=true`, `acquisition_freshness_verified=false`. Cached
poses can appear fresh. Omit it for the default clock-checked path, whose device
timestamp compatibility still needs operator verification. See below for the
full checks, watchdogs and limitations.

## 1. Inspect dependencies without connecting a device

From the SuperMimic repository root:

```bash
.venv/bin/python -m deployment.teleop_pico \
  --wbc-root "$HOME/Project/WholeBodyControl-PolicyTraining" --check
```

This reports interpreter/module availability, external GMR source/model/config
hashes, expected SDK APIs, simulator MJCF identity and grounding assumptions.
It does not import the SDK/GMR, call `xrt.init()`, start PC Service, or contact
Redis. A successful check command means the report was generated, **not** that
every dependency or device is ready; inspect each `available` field.

The external checkout is read in place; only its `data_processing` path is
imported, avoiding its unrelated `deployment` package. Nothing is copied or
installed automatically. Optional producer packages are listed separately:

```bash
uv pip install --python .venv/bin/python -r deployment/requirements-pico.txt
```

`xrobotoolkit_sdk` is a separately built native binding to your installed
XRoboToolkit PC Service. Use a build matching the producer's Python minor
version and CPU architecture; do not copy a CPython 3.10 ARM64 `.so` into the
Python 3.12 x86_64 environment. Existing SDK build instructions are in the
reference repositories. If a compatible SDK/GMR interpreter is already working,
invoke **this repository's** `python -m deployment.teleop_pico` with that
interpreter from the SuperMimic root; Redis isolates its ABI from the ONNX process.

Initial inspection on 2026-10-27 found PC Service's startup script at
`/opt/apps/roboticsservice/runService.sh`, but no SDK binding in SuperMimic's
`.venv` or the reference checkout's `.venv` / `.venv-gmr`. SuperMimic's `.venv`
also lacked `mink` and OpenCV. `.venv-gmr` had mink/daqp but still lacked the SDK.
This is a historical setup check, not the current installation status: the
operator subsequently installed the SDK and confirmed the timestamp-bypass
runtime above. No dependency/service was changed by the initial inspection;
rerun `--check` in the environment you intend to use.

Start/connect the Pico application and PC Service manually using your known
working setup. This program never starts, restarts or kills PC Service. It
prefers `get_body_timestamp_ns()`. Some Pico clients omit `Body.timeStampNs`
while supplying `Body.joints[i].t`: if the aggregate is initially zero, the
adapter chooses the newest positive joint timestamp once and pins that joint
for the producer's lifetime. It never substitutes the global/controller clock.
If the selected clock freezes, becomes zero, regresses, or has implausible units,
tracking still disarms. A different joint becoming newer does not change clocks.

`PICO DIAG` reports `selected_body_clock`, `selected_joint_index`, and
`selected_joint_name`. A permanently zero aggregate can therefore be compatible
when the selected joint nanosecond clock progresses. Snapshot checks compare
the entire joint timestamp array before/after poses in this fallback mode;
unchanged clocks do not become fresh just because an operator presses A.

For simulation debugging only, add `--ignore-sdk-timestamps` to the producer
command. This skips all SDK timestamp reads/checks (including diagnostic clock
reads) and `BodyClock` mapping, sampling poses at up to 50 Hz using host monotonic
time. Pose validation, Redis/control watchdogs and body-availability checks stay
enabled. **Cached/frozen SDK poses are indistinguishable from fresh samples if
`is_body_data_available()` stays true.** The startup warning and diagnostics
explicitly report `sdk_timestamps_ignored=true` and
`acquisition_freshness_verified=false`. Do not treat this as verified live tracking.

## 2. Start the model and local viewer

Use an existing local Redis (`redis-cli ping`); do not launch another instance
on an occupied port or clear databases used by other projects.

```bash
.venv/bin/python -m deployment.sim2sim \
  --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --stream-mode live --channel supermimic:pico
```

The model needs no motion file. After a discarded simulation-only warm-up and
full reset, it advertises `WAITING`. The warm-up initializes lazy inference and
physics work before control timing is monitored; it is not a received command.

An already running controller from an older command must be closed/restarted
with `--stream-mode live`. The default `csv` mode remains ACK-clocked file
playback and intentionally does not accept the live protocol.

## 3. Start the independent Pico/GMR producer

In another terminal, from the same repository root:

```bash
.venv/bin/python -m deployment.teleop_pico \
  --wbc-root "$HOME/Project/WholeBodyControl-PolicyTraining" \
  --human-height 1.6 --channel supermimic:pico --run
```

Set the actual operator height, not an arbitrary copied default. Both processes
accept `--redis-host`, `--redis-port`, `--redis-db`, and `--channel`; they must
match. Only loopback Redis hosts are accepted. Pico-to-PC networking stays in
XRoboToolkit, not an exposed Redis server.

- Right controller **A rising edge**: arm and fill a **new** sample buffer.
- Right **A** again: stop/pause the session, clear buffer, freeze simulation.
- Left controller **X**: stop and exit the producer; takes precedence over A.
- Viewer **Space** pauses/disarms live input. Unpause, then press A again to
  rearm from fresh samples. Viewer **R** stops the session. G/F toggle ghost/follow.

After stop, time regression, tracking loss or receiver failure, no automatic
rearming occurs. Release/repress A once valid body tracking is restored.
Restarting a session explicitly resets the **simulated** robot to the buffered
reference initial state and clears policy history; subsequent target updates
never teleport physics. This is not a real-robot startup strategy.

## 4. Coordinate, retargeting and timing conventions

- Raw SDK bodies are `[x,y,z,qx,qy,qz,qw]`; poses and body timestamps are read
  between matching body-clock samples, rejecting torn snapshots.
- The existing WBC `coordinate_transform_unity_data()` is reused (wxyz output),
  not SONIC's different three-point conversion. SDK units must be metres.
- GMR uses `src_human="xrobot"`, `tgt_robot="unitree_g1"`, solver `daqp`.
  Its `offset_to_ground=True` behavior includes a **0.1 m** foot target offset.
  This is retained/logged for reproducibility, not automatically calibrated to
  the training MJCF. Inspect foot/root tracking in the viewer before changing it.
- GMR hinges are mapped by name into the 29-joint contract. Every outgoing pose
  must satisfy both GMR and simulator MJCF ranges; errors disarm rather than clip.
  `--mjcf` on producer must identify the same robot XML used by the controller.
- Initial root XY and yaw are recentered once per session; height is preserved.
  `--no-recenter` preserves the original GMR world frame. Joint zero positions,
  observation scales, policy action scale and PD gains are not redefined.

[deployment.teleop.reference.BufferedReference](../teleop/reference.py) applies
source-dt exponential filtering (`--filter-tau .03`),
position/joint interpolation and quaternion SLERP on an exact 20 ms grid. It
computes joint/world linear velocities and SO(3) world angular velocities after
resampling. Every emitted row has a measured derivative guard sample beyond it;
no endpoint is extended into a claimed real future.

For K=10 the minimum startup span is `K*.02 + jitter_margin + .02`: with default
40 ms jitter margin, **260 ms** of measured data must first be available.
Acquisition, IK, causal filtering and transport add latency; startup span is not
the end-to-end response time. The producer sends bounded overlapping chunks;
the receiver advances a delayed reference cursor independently at 50 Hz.

## 5. Short underrun versus failure

In `PLAYING`, each 20 ms control tick constructs fresh robot observations and
history and runs ONNX; receipt of a packet does not itself cause a physical step.
At 2 ms physics dt, each tick runs ten explicit effort-clipped PD substeps.

When the next complete lookahead window is missing, `HOLDING` reuses the last
**whole reference window**, not the last action. The reference cursor stops but
new feedback/history/actions/physics continue for a bounded interval. Default
limits are:

- `--hold-timeout .12`: maximum short reference-window hold.
- `--reference-timeout .5`: source acquisition age/progress timeout.
- `--max-reference-delay 1`: refuse excessive backlog instead of catch-up replay.
- `--max-control-lateness .15`: freeze on an excessively slow control/render tick.

These are simulation diagnostics, not hard-real-time guarantees. Minor overruns
slow wall-time simulation without skipping physics; excessive stalls disarm.
Redis disconnect ends the runner after recording the failure; restart it and
rearm the producer. Long source outages require a fresh session even if packets
continue to be published with newer publication times.

## 6. Recording and verification

Viewer displays state, source age, reference delay, buffered rows and measured
control frequency. `--record` NPZ adds `source_sequence`, `source_time`,
`reference_time` (row-zero time of the window used for that tick), and `holding`.
Actual/reference positions are recorded after physics at the window's next row.
JSON retains live status, held-step count and maximum tick time; the existing
policy/motion-model hashes and modeling differences remain applicable.

Hardware-free tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  deployment/tests/test_teleop_reference.py deployment/tests/test_teleop_pico.py \
  deployment/tests/test_live_reference.py -q

SUPERMIMIC_ONNX=checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest deployment.tests.test_live_integration -v
```

The opt-in integration test uses **recorded poses as a simulated acquisition
source**, a temporary isolated Redis port, actual ONNX inference and MuJoCo.
It verifies independent ticks, overlapping buffer consumption, bounded hold,
timeout freezing, replay rejection and NPZ recording. It never initializes the
XRoboToolkit SDK. Temporary services and outputs are cleaned on exit.

Live Pico/GMR tracking has **not** completed qualification. After SDK installation,
operator logs showed aggregate body timestamp zero with progressing per-joint
timestamps; the adapter now supports a pinned joint-clock fallback for that
client format. The operator confirmed running with `--ignore-sdk-timestamps`,
not the default clock-checked fallback or complete hardware tracking quality.
The fallback still needs operator verification with the device. Synthetic
clocks, mocked SDK tests and recorded-pose replay do not
establish actual Pico latency, IK quality, calibration or full-motion tracking
performance. No physical robot was accessed.
