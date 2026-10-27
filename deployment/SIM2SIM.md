# ONNX policy visualization in standalone MuJoCo

This workflow follows the model/motion separation in
`WholeBodyControl-PolicyTraining`: export a checkpoint once, then select or
change the reference motion at simulation time. The ONNX graph never embeds a
walk/dance clip. The default workflow loads ONNX and the simulator first, then
waits for a separate Redis reference producer. The controller never opens the
sender's motion file. A direct-file mode is also available for offline comparisons.

The solid robot is driven by ONNX actions in real MuJoCo physics. The translucent
blue robot is reference-only geometry; it cannot exert forces or collide. This
is not a recording of the original Isaac Lab evaluator and not a hardware test.
No SDK2, DDS, robot interface, GPU, or C++ build is needed for this workflow.
The separate [DDS deployment bridge](SIMULATION.md) remains available unchanged.

## Dependencies

Run from the repository root using `.venv/bin/python`. NumPy, torch, ONNX,
ONNX Runtime CPU, MuJoCo and the Redis client are listed in `deployment/requirements.txt`:

```bash
uv pip install --python .venv/bin/python -r deployment/requirements.txt
```

The GUI needs a working desktop OpenGL display. Video additionally needs ffmpeg.
The robot assets must already exist at `data/g1_description/g1_29dof.xml` and
motions at `data/lafan1_g1/*.csv`. Nothing is downloaded during simulation.

## 1. Export only the policy

```bash
.venv/bin/python -m deployment.export_policy \
  --checkpoint checkpoints/1026_baseline_lafan/ckpt_020000.pt \
  --output checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --parity-samples 12
```

No `--motion` or `--mjcf` is required. The output must not already exist; export
will not overwrite it. Only the checkpoint's `pi` and frozen normalizers are
exported, using safe `weights_only=True` checkpoint loading and FP32 inference.
The exporter checks the graph and compares nonzero inputs with ONNX Runtime
when installed; a skipped check is explicitly reported rather than called a pass.
Dimensions and checkpoint SHA256 are embedded under the ONNX `supermimic` metadata
key. Default shapes are `obs_hist [1,10,99]`, `goal [1,730]`, `action [1,29]`;
history and goal horizon are read from the checkpoint, not hardcoded in the runner.

The old `--motion ... --mjcf ... --output BUNDLE_DIRECTORY` invocation still
creates a complete reference bundle for the C++/DDS deployment path.

## 2. Start the model first, then send a motion

Use a local Redis server (`redis-cli ping` should return `PONG`). If none is
running, start one in another terminal, bound only to loopback:

```bash
redis-server --bind 127.0.0.1 --port 6379 --protected-mode yes --save "" --appendonly no
```

Do not expose Redis on `0.0.0.0`. The receiver currently accepts local Redis
hosts only; future Pico network reception belongs in a separate local adapter.

**Terminal 1: load ONNX once and open the viewer. No motion argument:**

```bash
.venv/bin/python -m deployment.sim2sim \
  --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx
```

The window initially displays a frozen neutral robot and `WAITING`. It does not
step physics or infer actions without a valid reference. `--listen` is an
optional explicit spelling of this default mode.

**Terminal 2: send a motion, after the model is ready:**

```bash
.venv/bin/python -m deployment.send_motion \
  --motion data/lafan1_g1/walk1_subject1.csv \
  --hold-final-velocity
```

At the end the controller freezes and waits; it does not exit. Run the sender
again with another file, keeping Terminal 1 and its ONNX session alive:

```bash
.venv/bin/python -m deployment.send_motion \
  --motion data/lafan1_g1/dance1_subject1.csv \
  --hold-final-velocity
```

Motion/source selection and clip-tail conventions belong to the **sender**.
`--seconds 10` on the sender limits that playback to 10 simulated seconds. A
fresh start explicitly resets the simulated robot and policy history; ordinary
frame updates never teleport the robot. This reset is simulation-only, not a
real-robot startup procedure.

Both processes accept `--redis-host`, `--redis-port`, `--redis-db` and `--channel`.
Defaults are `127.0.0.1:6379`, DB 0, channel `supermimic:g1`; all must match.
Use a different channel to isolate multiple experiments. A short owner lease
prevents two senders or two controllers from owning the same channel.

Space pauses the controller and file sender together. **R stops the current
stream session**; start another sender to reset and resume. G/F/Esc keep their
usual meaning. The [ / ] keys apply only to direct-file mode below.

The sender waits for an acknowledgement **after** each 20 ms physics transition,
so slow rendering does not skip CSV reference frames. This is deliberately
simulation-clocked playback, not a hard-real-time teleoperation guarantee.
Missing references (default timeout 0.5 s), invalid values, wrong dimensions,
future timestamps and replayed sequences cannot continue driving physics.
After timeout, start a **new session**; an old stream cannot silently resume.
Stop/timeout freezes simulation, not a physical robot safety controller.

`--video`, `--record` and `--metrics` also work in stream mode. Waiting/paused
wall time is not inserted into the video; NPZ records include source sequence
and segment IDs. `--seconds` on the controller limits total active simulated
time across sessions; otherwise the process stays alive until closed.

See [REFERENCE_STREAM.md](REFERENCE_STREAM.md) for the file packet contract.
For the independent buffered Pico/GMR producer and live control clock, use
[PICO.md](PICO.md). Pico requires `--stream-mode live`; CSV defaults remain unchanged.

## 2b. Optional direct-file mode

```bash
.venv/bin/python -m deployment.sim2sim \
  --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --motion-dir data/lafan1_g1 \
  --clip walk1_subject1 \
  --hold-final-velocity
```

For one clip, replace `--motion-dir ... --clip ...` with
`--motion data/lafan1_g1/walk1_subject1.csv`. The same ONNX works with every
compatible CSV. The model stays loaded when switching clips.

Keyboard controls (viewer focused):

- **Space**: pause/resume both physics and reference time.
- **R**: reset the current clip and resume.
- **[ / ]**: previous/next CSV in sorted order, resetting the entire controller.
- **G**: toggle the translucent reference.
- **F**: toggle camera following; MuJoCo mouse controls remain available.
- **Esc** or close the window: quit and flush requested outputs.

At clip end, the GUI stops physics and holds the final displayed state until
reset or clip selection. There is no silent replay. `--loop` explicitly enables
full-state resets at the end. `--paused` starts with the initial pose frozen.
Reset/switch clears history and previous action, restores reference frame 0
including velocities, and resets the reference/physics clocks together.

Reference XY is anchored to the simulated robot by default, as in the existing
highlight renderer, to make local pose comparison easier. This does not alter
physics or the goals. Add `--global-reference` to reveal world-position drift.
RMSE always measures the true world positions, regardless of ghost placement.

### Clip-tail compatibility

Job 74284's checkpoint predates the upstream stationary-tail goal fix. Use
`--hold-final-velocity` for this checkpoint to reproduce its old goal convention.
For checkpoints trained with the current `MotionLib.goal`, omit that flag: future
poses beyond clip end are held with **zero** linear/angular/joint velocity, while
the actual recorded endpoint velocity is retained. This changes runtime goals,
not the ONNX network, and only affects the last goal horizon of the clip.

## 3. Record the actual ONNX closed loop

```bash
mkdir -p .cache/deployment/sim2sim
.venv/bin/python -m deployment.sim2sim \
  --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
  --motion data/lafan1_g1/walk1_subject1.csv \
  --hold-final-velocity --seconds 10 \
  --video .cache/deployment/sim2sim/walk.mp4 \
  --record .cache/deployment/sim2sim/walk.npz \
  --metrics .cache/deployment/sim2sim/walk.json
```

All output parents must already exist, output paths must be distinct, and existing
outputs are refused. `--seconds` measures total **simulated** time, excluding
pauses. Without it, headless runs end at the clip end; GUI runs wait for the user.

`--headless --no-realtime` runs faster without a window. For offscreen video with
a supported NVIDIA EGL driver, set `MUJOCO_GL=egl`. On a machine without working
EGL, a software/virtual-display fallback (when Xvfb and Mesa are installed) is:

```bash
xvfb-run -a -s "-screen 0 1280x720x24" \
  env MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 \
  .venv/bin/python -m deployment.sim2sim \
    --policy checkpoints/1026_baseline_lafan/ckpt_020000.onnx \
    --motion data/lafan1_g1/walk1_subject1.csv \
    --hold-final-velocity --headless --no-realtime --seconds 10 \
    --video .cache/deployment/sim2sim/walk-software.mp4
```

Video is sampled at 25 FPS by simulation time, not once per policy step or once
per GUI render. Slow rendering/inference changes wall time, not playback speed.
`--fps` accepts positive divisors of 50. The NPZ is a different format from
`evaluate.py --record`: it contains control-rate `time`, `segment`, `clip`,
`frame`, `root_pose_wxyz`, `joint_position`, `reference` (71D), and `action`.
Each explicit reset/clip change increments `segment`, so discontinuities are not
silently joined. Load with `np.load(..., allow_pickle=False)`.

JSON contains per-segment joint/root RMSE, minimum root height, torque-clipping
counts, inference p95, policy/motion/MJCF hashes, and timing/modeling assumptions.
It deliberately reports `success_evaluated=false`; no-crash or a short attractive
video is not a full 40-clip evaluation or a safe-to-deploy claim.

## Modeling differences and validation

- Inference is CPU FP32 at 50 Hz, versus bf16 in the original GPU evaluation.
- Physics is explicit torque-limited PD at 2 ms, versus the training simulator's
  implicit PD and 5 ms substeps. Gains/action scales and named joint ordering come
  from the deployment contract; raw actions are not clipped to `[-1,1]`.
- Self-collision remains disabled. The standalone runner preserves original
  non-colliding visual meshes and the foot contact spheres, rather than turning
  decorative meshes into additional contacts. The legacy DDS bridge defaults
  are unchanged.
- Future goals and observations use the existing SuperMimic contract. No extra
  joint-velocity/gyro scaling, reference teleportation of the physical robot, or
  reference-driven physics stepping is applied.

Local verification with the actual `ckpt_020000.pt`: standalone export passed
12-sample ORT parity (maximum absolute action error `1.90735e-6`). A 10-second
walk rollout completed 500 control steps; 25 FPS output contained exactly 250
frames over 10 seconds. Joint-position RMSE was about 0.0875 rad, with zero motor
torque clips; this is a single short diagnostic, not a claim about every motion.
The passive viewer, ghost scene and text overlay also ran under Xvfb. Physical
desktop keyboard/mouse usability still needs operator confirmation.

Focused tests can be run without a display or DDS:

```bash
python -m pytest deployment/tests/test_sim2sim.py \
  deployment/tests/test_export_model.py deployment/tests/test_contract.py -q
```
