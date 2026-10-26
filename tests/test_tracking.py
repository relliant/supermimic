"""Exercise the tracking lifecycle without Isaac Lab, a GPU or cloud credentials."""

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from supermimic.tracking import add_swanlab_args, track_experiment


def parse_tracking_arguments(*options: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_swanlab_args(parser)
    return parser.parse_args(options)


class TrackingTests(unittest.TestCase):
    def test_training_uses_requested_destination_and_flushes(self):
        arguments = parse_tracking_arguments()
        swanlab = MagicMock()
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(sys.modules, {"swanlab": swanlab}):
            run_directory = Path(temporary_directory) / "training-run"
            with track_experiment(arguments, log_dir=str(run_directory), job_type="training") as experiment:
                self.assertIs(experiment, swanlab.init.return_value)
                swanlab.finish.assert_not_called()
            initialization = swanlab.init.call_args.kwargs
            self.assertEqual(initialization["workspace"], "relliant")
            self.assertEqual(initialization["project"], "supermimic_g1")
            self.assertEqual(initialization["experiment_name"], "training-run")
            self.assertEqual(initialization["mode"], "online")
            self.assertEqual(initialization["config"], vars(arguments))
            self.assertEqual(Path(initialization["logdir"]), run_directory / "swanlab" / "training")
            self.assertTrue(Path(initialization["logdir"]).is_dir())
        swanlab.finish.assert_called_once_with()

    def test_disabled_mode_does_not_import_sdk_or_create_logs(self):
        arguments = parse_tracking_arguments("--swanlab_mode", "disabled")
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(sys.modules, {"swanlab": None}):
            run_directory = Path(temporary_directory) / "unused"
            with track_experiment(arguments, log_dir=str(run_directory), job_type="training") as experiment:
                self.assertIsNone(experiment)
            self.assertFalse(run_directory.exists())

    def test_evaluation_is_separate_and_flushes_on_interruption(self):
        arguments = parse_tracking_arguments("--swanlab_mode", "offline")
        swanlab = MagicMock()
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(sys.modules, {"swanlab": swanlab}):
            run_directory = Path(temporary_directory) / "training-run"
            with self.assertRaises(KeyboardInterrupt):
                with track_experiment(arguments, log_dir=str(run_directory), job_type="evaluation"):
                    raise KeyboardInterrupt
            initialization = swanlab.init.call_args.kwargs
            self.assertEqual(initialization["experiment_name"], "training-run-eval")
            self.assertEqual(initialization["job_type"], "evaluation")
            self.assertEqual(initialization["mode"], "offline")
            self.assertEqual(Path(initialization["logdir"]), run_directory / "swanlab" / "evaluation")
        swanlab.finish.assert_called_once_with()

    def test_destination_name_and_resume_can_be_overridden(self):
        arguments = parse_tracking_arguments(
            "--swanlab_workspace", "other-workspace",
            "--swanlab_project", "other-project",
            "--swanlab_name", "custom-experiment",
            "--swanlab_run_id", "existing-experiment",
        )
        swanlab = MagicMock()
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(sys.modules, {"swanlab": swanlab}):
            with track_experiment(arguments, log_dir=temporary_directory, job_type="evaluation"):
                pass
        initialization = swanlab.init.call_args.kwargs
        self.assertEqual(initialization["workspace"], "other-workspace")
        self.assertEqual(initialization["project"], "other-project")
        self.assertEqual(initialization["experiment_name"], "custom-experiment")
        self.assertEqual(initialization["id"], "existing-experiment")
        self.assertEqual(initialization["resume"], "allow")

    def test_missing_sdk_has_actionable_error(self):
        arguments = parse_tracking_arguments()
        with patch.dict(sys.modules, {"swanlab": None}):
            with self.assertRaisesRegex(RuntimeError, "--swanlab_mode disabled"):
                with track_experiment(arguments, log_dir="unused", job_type="training"):
                    self.fail("Tracking should not start without the SDK")


if __name__ == "__main__":
    unittest.main()
