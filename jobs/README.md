# Running SuperMimic on the Mondo H100 cluster

Use the authenticated H100 development pod (`ssh mondo-aliyun-h100`), not the local
desktop's separate md_ak session. No token transfer or new login is needed there.

## Code and environment

- Shared code: `/data/nas_ray/home/sia.luo/code/supermimic/a1508d4`.
- Base commit: `a1508d42c9d1699a52e8233ab16d08c488441ab9`.
- Runtime changes: `local.patch` in the shared code directory, Git blob
  `d02d89195ddaabc5998f56b358412b9c06a4ff3c`. Only the reusable host setup and data
  download tools are patched; training algorithm and hyperparameter defaults are unchanged.
- Environment: `<code>/IsaacLab/.venv`, using Isaac Lab at
  `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` and its frozen uv lock.
- Assets: `<code>/data`; caches: `<code>/.cache` and the user's shared uv cache.
- Formal run output: `/data/nas_ray/home/sia.luo/experiments/supermimic/lafan1_main`.

## Job sequence

`supermimic_h100.yaml` combines the following smoke, training, and evaluation tasks
into one sequential Job. Later tasks remain BLOCKED until their predecessor succeeds.

1. `supermimic_h100_smoke.yaml`: synchronize the environment, prepare assets, check
   FK against the simulator, then run 510 iterations to exercise policy updates
   after the 500-iteration world-model warm-up. Each invocation uses a fresh smoke
   output directory. A successful run writes `<code>/.cache/smoke-passed` containing
   the runtime patch hash.
2. `supermimic_h100_train.yaml`: requires that matching smoke marker, trains for
   20,000 iterations with 4,096 environments on one GPU, then evaluates all
   checkpoints. It refuses to overwrite an existing training log. Target return
   and failure counts are reported separately from execution success.

Both jobs prefer `rx-brain.aliyun_sh_h100`. Do not switch to default merely because
an older local CLI does not recognize dotted queue names. md_ak 2.1.1 on the devpod
has that client-side limitation; its authenticated Central submission helper
accepts the configuration without changing the queue:

```python
from datetime import datetime
from importlib.metadata import version
from md_ai_kit.cli import _read_config_to_dict, _submit_to_central, get_user_id

configuration = _read_config_to_dict("jobs/supermimic_h100.yaml")
configuration["md_ai_kit_version"] = version("md_ai_kit")
configuration["run_id"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_submit_to_central(configuration, get_user_id())
```

This uses the same authenticated submission helper as `md_ak submit`; no credentials
are printed, copied, or embedded in configuration.

## Current submission

- Active pipeline: job **73872**, submitted 2026-10-26.
- Web: https://ak.mondorobotics.com/jobs/73872
- Smoke **88010**: RUNNING, currently bootstrapping the uv environment.
- Training **88011** and evaluation **88012**: BLOCKED on their predecessors.
- Logs: `md_ak logs --job 73872 --task-index 0 --tail 12000` (use index 1 for training).
- Status: `md_ak detail --job 73872`.
- The earlier standalone smoke **73867** was stopped; it did not reach training.

The cluster name is not a physical GPU model guarantee. The allocated smoke device
reports **NVIDIA L20Z**, 81,559 MiB, compute capability **8.9**, driver **580.105.08**.
This is not an H100 GPU (compute capability 9.0), even though it was assigned from
`rx-brain` in `aliyun_sh_h100`. This run uses the requested cluster and must not be
reported as a physical-H100 performance reproduction.

The CUDA 13 packages need a compatible NVIDIA driver. Driver compatibility,
package-index access, asset conversion and actual GPU execution are verified by
the smoke job rather than assumed from the image name.
