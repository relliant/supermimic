# Independent reference producer protocol (simulation only)

`deployment.sim2sim --policy POLICY.onnx` starts the persistent receiver and
viewer. `deployment.send_motion --motion CLIP.csv` is an independent producer.
The receiver never reads CSVs or loads a new ONNX at clip changes. Both use
local Redis by default; this is a separate protocol from the other repository's
Redis fields, which represent different policy observations and robot layouts.

## Version-2 buffered Pico live mode

Use `deployment.sim2sim --stream-mode live` with the separate
`deployment.teleop_pico --wbc-root ... --run` producer. Full setup and limitations
are in [PICO.md](PICO.md). File/CSV mode remains version 1 below; selecting live
mode does not change the ONNX model or accept the other repository's messages.

Version 2 uses the same schema name and `control_dt=0.02`, but carries:

- `controller_id`, `session_id`, `sequence`, `state`, `name`, `sent_at`;
- `source_sequence`: selected SDK body clock in nanoseconds (aggregate body
  timestamp, or one pinned positive body-joint timestamp when aggregate is absent);
- `source_time`: body acquisition clock mapped to host Unix seconds;
- `reference_time`: delayed time of the first reference row;
- `reference`: K+1..128 overlapping, complete 71D rows on a 20 ms timeline,
  or null for stop. No `horizon_mode` field: all live rows are buffered measurements.

The optional simulation-debug `--ignore-sdk-timestamps` path instead uses host
sampling time and explicitly reports `acquisition_freshness_verified=false`;
it cannot distinguish cached SDK poses from fresh acquisitions. See
[Pico timestamp limitations](PICO.md) before using that bypass.

Publication, source freshness and reference time are distinct. Refreshing a
publication cannot refresh an old body sample. Rows must end no later than the
latest observed acquisition time. Packets are limited to 256 KiB. Start/reset is
ACKed before the first control tick; ordinary receipt/sequence changes do not
clock physics. Overlapping rows are merged on a fixed session grid, duplicates
and out-of-order packet sequences are ignored, discontinuities require rearming.

The independent 50 Hz control tick selects current + K future rows, then runs
ten 2 ms physics substeps with fresh feedback/history/action inference. A short
underrun holds the last complete **reference window** and reference cursor,
while control continues. After the hold budget (default 0.12 s), source timeout
(0.5 s), excessive delay (1 s), or excessive control/render latency (0.15 s),
the simulation freezes and requires a new session. These are not hardware
safety guarantees. Viewer live pause disarms and clears the reference cache;
resuming requires a newly filled source buffer and explicit A rearm.

Live status includes `stream_mode`, source age, reference delay, buffer depth,
held/control-step counts and control frequency. Recording adds `source_sequence`,
`source_time`, `reference_time`, and `holding`. `reference_time` is the time of
window row zero; the actual state/reference recorded after physics uses row one.

The receiver performs a discarded simulation-only warm-up and full reset before
advertising readiness; this is not a source packet or a counted control tick.

## Version-1 CSV/ACK mode

## Transport and lifecycle

Defaults: `127.0.0.1:6379`, DB 0, channel `supermimic:g1`.

- `<channel>:status`: JSON heartbeat, expiring after 2 seconds. Includes
  `controller_id`, `goal_frames`, `control_dt=0.02`, `state`, `session_id`,
  `sequence`, `updated_at` (Unix seconds), and `paused`.
- `<channel>:reference`: the latest reference packet, not a growing queue.
- `<channel>:controller` and `<channel>:publisher`: 3-second ownership leases,
  renewed and deleted only by their owners. Use a different channel for parallel
  experiments; never clear a shared Redis database to reset this program.

The file sender reads the controller's advertised horizon; it does not assume
K=10. It publishes one packet and waits until status acknowledges that sequence
**after** inference and one 20 ms physics transition. A paused controller keeps
heartbeating; the sender refreshes its pending packet but never advances it.
This ensures file playback uses simulation time even on a slow renderer.

Each sender invocation uses a new session ID. Start sequence zero explicitly
resets the simulated full state, history, previous action and local clock.
Following frames retain physical state/history and change only reference inputs.
An ordinary `stop` acknowledges without stepping, freezes the simulator, and
keeps the model loaded. Subsequent frames from that session are rejected.

Missing/expired input freezes physics after the configured reference timeout
(default 0.5 seconds). Invalid packets also stop the session; repeated sequences
never refresh its watchdog or execute another physics step. Recovery requires
a fresh start/session ID. A changed controller ID invalidates old producers and
cached packets. No automatic robot recovery or real-hardware safety is implied.

## Packet contract

Use `deployment.streaming.protocol.ReferencePacket` and `.encode()` to create
bounded JSON. The receiver uses `decode()` and rejects unknown fields, invalid
dimensions, nonfinite values, bad quaternion norms, negative/out-of-order
sequences, expired packets, or timestamps over 0.5 seconds in the future.
Maximum encoded size is 256 KiB. Local processes use the same host clock.

Required fields:

- `schema="supermimic.reference"`, `version=1`, `control_dt=0.02`.
- `controller_id`: from current status; `session_id`: a new UUID per playback.
- `sequence`: zero for start, strictly increasing for frames/stop.
- `sent_at`: Unix seconds; use the time of the newest source sample, not merely
  the time a stale cached sensor sample was republished.
- `state`: `start`, `frame`, or `stop`.
- `name`: display label, never a path opened by the controller.
- `horizon_mode`: `recorded`, `hold`, or `predicted`, retained in diagnostics.
- `reference`: float32-compatible array `[K+1,71]` (or null for stop).

Row zero is the current desired state; rows 1..K are desired states at
`t+20ms .. t+K*20ms`. Every row uses SI units and this layout:

1. root world position (3);
2. normalized root quaternion **wxyz** (4);
3. G1 29 joint positions in `deployment.common.contract.G1_JOINT_NAMES` order;
4. corresponding 29 joint velocities;
5. root world linear velocity (3);
6. root world angular velocity (3).

The receiver constructs the policy's relative, heading-frame goals and current
reference-orientation observation itself. Producers must not send ONNX actions,
raw Pico poses, another repository's preprocessed observations, or a 730D vector
with an unspecified convention. Quaternion xyzw, body-axis root velocities,
degree-valued joints and alternate motor order are incompatible.

## Pico integration boundary

The implemented `teleop_pico` adapter uses the existing XRoboToolkit PC Service
and external GMR, retargets into this robot's 29-DoF coordinates, checks named
joint limits, estimates velocities, and buffers a real future window. It uses
version 2, not the version-1 CSV timestamp refresh described below. No independent
future predictor or physical robot transport is implemented.

SuperMimic is noncausal with respect to its reference: K=10 means a 200 ms future
horizon. Live Pico tracking does not provide true future samples. Possible
adapter strategies include a buffered/delayed window, explicit prediction, or
zero-velocity pose holding. Each changes latency or the training input
distribution and must be tested; enabling the protocol alone does not establish
teleoperation quality.

For a simple **hold** window, repeat the current pose for K future rows and zero
the future rows' velocity columns (36..70), retaining measured current velocities
in row zero. Set `horizon_mode="hold"`. A predictor instead supplies its own
consistent position/orientation/velocity rows and sets `"predicted"`. CSV playback
uses `"recorded"` and preserves the selected clip-tail convention.

Reusable producer helpers live in `deployment.streaming.motion.MotionPublisher`:
`acquire()`, `wait_ready()`, `send_and_wait(packet)`, and
`close(interrupted=...)`. These manage the owner lease, controller identity,
pause-aware acknowledgements and a stop on interruption. For true live input,
do not use their file-playback timestamp refresh as a substitute for checking
the upstream sensor timestamp: stop when Pico tracking becomes stale. A custom
latest-sample producer may publish newer sequences without waiting for ACKs;
unconsumed intermediate samples are intentionally overwritten, not queued.

Viewer recordings include `reference_sequence`, per-session `segment`, actual
state, desired state and action. The live source label and window convention
are diagnostic metadata, not proof of a motion file's provenance.

## Verification performed

An isolated temporary Redis server and the actual exported `ckpt_020000.onnx`
were used with separate controller/sender processes. The controller started
without motion, accepted 50 walk steps then 50 dance steps from two independent
sender invocations, acknowledged both stops, and retained the same controller ID.
Two further adapter-style windows tested `predicted` and `hold` modes, stream
expiration, frozen physics, and fresh-session recovery. All 102 actual control
steps and four segments were verified from the NPZ/JSON records. Temporary test
processes, server, scripts and outputs were removed; no existing Redis service
was modified. This does not validate real Pico data or hardware behavior.
