"""Shared SwanLab configuration for training and standalone evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path


def add_swanlab_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swanlab_workspace", default="relliant")
    parser.add_argument("--swanlab_project", default="supermimic_g1")
    parser.add_argument("--swanlab_name", default=None, help="experiment name; defaults to the run directory name")
    parser.add_argument("--swanlab_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--swanlab_run_id", default=None, help="resume a SwanLab experiment by ID (independent of checkpoint resume)")


@contextmanager
def track_experiment(arguments: argparse.Namespace, *, log_dir: str, job_type: str):
    """Flush the experiment on exit, without importing SwanLab when disabled."""
    if arguments.swanlab_mode == "disabled":
        yield None
        return

    try:
        import swanlab
    except ModuleNotFoundError as error:
        if error.name != "swanlab":
            raise
        raise RuntimeError(
            "SwanLab is not installed. Install docker/requirements.txt in the training "
            "environment, or pass --swanlab_mode disabled to keep only local CSV logs."
        ) from error

    run_directory = Path(log_dir).resolve()
    experiment_name = arguments.swanlab_name or run_directory.name
    if job_type == "evaluation" and arguments.swanlab_name is None:
        experiment_name += "-eval"
    tracking_directory = run_directory / "swanlab" / job_type
    tracking_directory.mkdir(parents=True, exist_ok=True)
    experiment = swanlab.init(
        workspace=arguments.swanlab_workspace,
        project=arguments.swanlab_project,
        experiment_name=experiment_name,
        job_type=job_type,
        config=vars(arguments).copy(),
        logdir=str(tracking_directory),
        mode=arguments.swanlab_mode,
        id=arguments.swanlab_run_id,
        resume="allow" if arguments.swanlab_run_id else None,
    )
    try:
        yield experiment
    finally:
        swanlab.finish()
