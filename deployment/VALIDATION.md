# Deployment validation receipt

Date: 2026-10-26. Base repository HEAD during implementation:
`a1508d42c9d1699a52e8233ab16d08c488441ab9`, with uncommitted deployment additions
and preexisting/parallel changes. No commit was created. No remote training files
or robot files were changed. SONIC's reference checkout was read-only.

## What was actually tested

Environment: Linux x86_64, GCC 11.4, CMake 3.22.1, Eigen 3.4, nlohmann-json
3.10.5, OpenSSL 3.0.2 headers, Python 3.11.15, torch 2.7.0+cu126, NumPy 1.26.0,
ONNX 1.22.0, ONNX Runtime CPU 1.24.4 and MuJoCo 3.11.0. SDK2 C++ libraries were
referenced from the external SONIC tree. Python DDS validation used official
Unitree SDK2 Python source and its required CycloneDDS 0.10.2 binding. These are
validation versions, not a claim that the training environment was installed.

- Native **supermimic_runtime**, **supermimic_offline**, **supermimic_g1**,
  **test_runtime** and **test_safety** compiled successfully, without CUDA or
  TensorRT linkage in the new runtime.
- `ctest --test-dir .cache/deployment/build --output-on-failure`: **2/2 passed**.
- `python -m pytest deployment/tests -q`: **81 passed, 1 skipped**. The skip is
  intentional opt-in native DDS integration, not an unimplemented unit test.
- `test_safety BUNDLE`: additional bundle-dependent tests passed for actual-dt
  slew limiting, applied-action feedback, NaN/Inf rejection, torque envelope and
  hardware-profile refusal. State watchdog, invalid quaternion, temperature,
  signed velocity/torque and motor-fault tests also passed.
- Opt-in native/DDS suite: **6 passed in 29.29 s** on the final build. Tests were
  sequential and bound only to `lo`, domains 42-45. Monitor sent zero commands;
  synthetic policy commands advanced MuJoCo; state loss stopped native control;
  command loss triggered bridge damping/stale handling; unvalidated hardware
  configuration failed before any DDS participant was initialized. A regression
  additionally repeated controller teardown five times while states were still
  being published.
- Python syntax compilation and IDE diagnostics were clean. PyTorch's legacy
  ONNX logging emits deprecation warnings; no unsafe loading fallback is used.

The bridge tests exposed and fixed a MuJoCo 3.11 crash: walking `MjSpec`
ancestry beyond `worldbody` accesses an unsafe wrapper. The regression compiles
100 temporary models sequentially and stops ancestry traversal at worldbody.

## Cross-language numerical and timing results

A deterministic **untrained synthetic** H=10, K=10 policy with initialized
synthetic normalization, 51-frame reference and 12 nontrivial parity samples was
exported. No learned checkpoint was substituted or automatically selected.

- Observations: maximum absolute difference **5.9604645e-08** across 1,188 values.
- History: **5.9604645e-08** across 11,880 values.
- Goals: **0** across 8,760 values.
- Actions: **6.9849193e-10** across 348 values; zero tolerance violations using
  `atol=1e-5, rtol=1e-4`.
- Full observation/history/goal/inference pipeline, one ORT thread, 20 warm-ups,
  300 measurements on this development machine: p50 **1.542 ms**, p95 **2.029 ms**,
  p99 **2.248 ms**, maximum **2.369 ms**, **0** overruns of 20 ms.

CPU scheduling and actual onboard conditions differ. Repeat on the robot's
computer and on the selected learned model before authorizing control. The
synthetic DDS fixture is a free-root sphere with 29 named hinge joints and a
zero-action policy: it checks transport and watchdogs, **not G1 dynamics or
learned motion tracking**.

## DDS shutdown race found and fixed

An early isolated run encountered an intermittent assertion in the external
CycloneDDS C++ teardown (`EntityDelegate::prevent_callbacks`, exit -6), after
tracking had completed. Merely dropping the channel allowed its last entity
reference to disappear on a DDS callback thread. Upstream 0.10.2's assertion
specifically describes this destructor-on-callback deadlock and recommends
explicit main-thread closure. The adapter now retains the channel and explicitly
closes the native reader on the controller thread before releasing it. The
final suite, including five live-publisher teardown repetitions, passed after
this correction. No retry masking, ignored exit codes or relaxed deadlines were
used. Still qualify the actual onboard SDK/library versions and physical
loss-of-command behavior; software process exit is not a safe handoff by itself.

## Remaining mandatory gates

- Select the actual trained checkpoint, G1 MJCF and first low-risk reference
  clip. Repeat export/parity on those inputs and retain their hashes.
- Run learned-policy closed-loop tests with the authentic G1 geometry, collisions,
  contact parameters and actuator model. Compare original bf16 evaluation with
  FP32 export and record explicit-PD/2-ms versus implicit-PD/5-ms differences.
- Review reference initial velocities and pose transitions; test all fault/exit
  paths. Disabled self-collision in the original simulator remains a known risk.
- Verify onboard x86_64 ABI/versions and full-loop latency under expected load.
- Verify IMU mounting/firmware conventions; current hardware adapter requires a
  calibrated rigid pelvis IMU and refuses articulated torso-IMU profiles.
- Review physical zeroes, motor order, gains, limits, payload, control ownership,
  bounded damping and real loss-of-command behavior with protection equipment,
  a trained operator and independent emergency stop.

No G1 SSH session, robot-facing DDS initialization, motor enable or hardware
closed-loop experiment was performed.

## Artifact disposition

Native binaries and the locally extracted dependency headers/shared-library
references remain under ignored `.cache/deployment/` for rebuilding and local
tests. No system packages or global Python environment were modified. Dependency
source/bindings needed by the reusable isolated tests remain cached.

One-off dependency-discovery downloads, package archives and standalone synthetic
export inputs/bundle were removed after recording the results. Integration
fixtures use temporary directories and reap subprocesses on every exit path.
No remote/NAS artifacts were created or cleaned in this deployment task; active
training and the user's parallel SwanLab changes were left untouched.
