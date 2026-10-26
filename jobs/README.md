# Running SuperMimic on the Mondo H100 cluster

Use the authenticated H100 development pod (`ssh mondo-aliyun-h100`), not the local
desktop's separate md_ak session. No token transfer or new login is needed there.

## Code and environment

- Shared code: `/data/nas_ray/home/sia.luo/supermimic` — the repository root itself, laid out
  exactly like the developer checkout (`<code>/supermimic/`, `<code>/tools/`, `<code>/jobs/`, …).
  There is no commit-named subdirectory and no `code/` level (it used to be
  `.../sia.luo/code/supermimic/a1508d4`). The commit is tracked by Git, not by the path, so a new
  commit does not require a new directory.
- Base commit: `264db00276b29d22ac2dd9ee161c148561740e4a`.
- Runtime changes: `local.patch` in the shared code directory, Git blob
  `1e88207760ebb3eee0fb9d5729f27bb8eb701e59` — the uncommitted diff of the developer checkout
  that the shared directory was synced from, with `jobs/` left out (that directory contains the
  assertions below, so including it would make the hash depend on itself). The diff only touches
  the host setup/data tooling and the deployment tree; training algorithm and hyperparameter
  defaults are unchanged.
- The shared directory is a byte-for-byte copy of the developer checkout's working tree at that
  commit (same files, same Git objects), produced with `tar` because the dev pod has no `rsync`.
  Re-sync it in place after committing locally; the path stays the same across commits, so the
  job YAMLs only need a new base commit and a new `local.patch` hash.
- Environment: `<code>/.venv` at the repository root, built by `tools/setup_uv.sh`, which keeps
  Isaac Lab at `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` and its frozen uv lock as the project
  and only redirects the environment's location (`UV_PROJECT_ENVIRONMENT`). Any earlier
  `<code>/IsaacLab/.venv` is stale after a code move (its paths are absolute) and has to be
  deleted. Do not move the directory while a job still uses it as `shared_workspace`.
- `tools/setup_uv.sh` is run manually, not by the smoke job. Changes outside `jobs/` change
  the `local.patch` blob hash, so the provenance assertions in the job YAMLs must be updated
  together with that patch.
- `local.patch` is not tracked by Git, so it has to be regenerated in the shared directory after
  a sync, and the resulting blob hash has to match the assertions in the job YAMLs:

  ```bash
  cd /data/nas_ray/home/sia.luo/supermimic
  git diff -- . ':(exclude)jobs' > local.patch
  git hash-object local.patch          # must print 1e88207760ebb3eee0fb9d5729f27bb8eb701e59
  git apply --reverse --check local.patch   # must succeed: the patch is already applied
  ```
- Assets: `<code>/data`; caches: `<code>/.cache` and the user's shared uv cache.
- Formal run output: `/data/nas_ray/home/sia.luo/experiments/supermimic/lafan1_main`.

## Job sequence

`supermimic_h100.yaml` combines the following smoke, training, and evaluation tasks
into one sequential Job. Later tasks remain BLOCKED until their predecessor succeeds.

1. `supermimic_h100_smoke.yaml`: reuse the already configured `<code>/.venv`, prepare assets, check
   FK against the simulator, then run 510 iterations to exercise policy updates
   after the 500-iteration world-model warm-up. Each invocation uses a fresh smoke
   output directory. A successful run writes `<code>/.cache/smoke-passed` containing
   the runtime patch hash.
2. `supermimic_h100_train.yaml`: requires that matching smoke marker, trains for
   20,000 iterations with 4,096 environments on one GPU, then evaluates all
   checkpoints. It refuses to overwrite an existing training log. Target return
   and failure counts are reported separately from execution success.

The jobs do not run `uv sync`, install packages, or change the prepared environment.
SwanLab logging is offline for smoke, training and evaluation; CSV logs, checkpoints
and local SwanLab records remain under each run directory without needing container credentials.

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

- Job **74284**, submitted 2026-10-26 at 21:45 UTC+8, uses
  `jobs/supermimic_h100.yaml` with one GPU in `rx-brain.aliyun_sh_h100`.
- Web: https://ak.mondorobotics.com/jobs/74284
- Smoke **88614** SUCCEEDED, including iteration 510 with policy and critic updates.
  Training **88615** is RUNNING; evaluation **88616** is BLOCKED until training succeeds.
  This is a smoke/startup record, not a completed training or learning-quality result.
- Shared code: `/data/nas_ray/home/sia.luo/supermimic`, base commit and runtime patch
  as recorded above. Formal output: `/data/nas_ray/home/sia.luo/experiments/supermimic/lafan1_main`.
- Status: `md_ak detail --job 74284`.
- Logs: `md_ak logs --job 74284 --task-index 0 --tail 12000` (index 1 for training).
- Jobs **73872** and **73867** were stopped before training; no old run is resumed.
- Job **74275** failed before simulation because its YAML used the unsupported
  `--headless` flag. The flag is removed from all tasks; this kit-less launcher defaults
  to no visualizer when none is selected.

The prepared `<code>/.venv` has been checked: torch `2.11.0+cu130`, mujoco `3.15.0`,
warp-lang `1.17.0` and swanlab `0.10.2` import successfully. GPU execution is checked
on the allocated worker by the smoke task, not inferred from the development pod.

The cluster name is not a physical GPU model guarantee. Job 74284's allocated smoke device
reports **NVIDIA L20Z**, 81,559 MiB, compute capability **8.9**, driver **580.105.08**.
This is not an H100 GPU (compute capability 9.0), even though it was assigned from
`rx-brain` in `aliyun_sh_h100`. This run uses the requested cluster and must not be
reported as a physical-H100 performance reproduction.

The CUDA 13 packages need a compatible NVIDIA driver. Driver compatibility,
package-index access, asset conversion and actual GPU execution are verified by
the smoke job rather than assumed from the image name.
