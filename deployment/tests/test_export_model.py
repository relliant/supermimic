"""Motion-independent exporter regressions, runnable with unittest or pytest."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from deployment.export import policy as export_policy
from deployment.common.contract import sha256_file


@unittest.skipUnless(importlib.util.find_spec("onnx"), "optional onnx package is not installed")
class StandaloneExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.checkpoint_directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.checkpoint_directory.name) / "checkpoint.pt"
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(371)
            cls.policy = export_policy.Policy(history_length=3, goal_frames=2).eval()
            for normalizer in (cls.policy.obs_norm, cls.policy.goal_norm):
                normalizer.mean.copy_(torch.linspace(-0.4, 0.7, normalizer.mean.numel()))
                normalizer.var.copy_(torch.linspace(0.0, 1.2, normalizer.var.numel()))
                normalizer.count.fill_(64)
            torch.save({
                "pi": cls.policy.state_dict(), "wm": {"irrelevant": torch.ones(1)},
                "args": {"hist": 3, "goal_frames": 2, "amp": "bf16"},
            }, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.checkpoint_directory.cleanup()
        torch.set_num_threads(cls.previous_thread_count)

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.output = self.directory / "policy.onnx"

    def test_standalone_graph_metadata_and_real_runtime_parity(self):
        import onnx

        with mock.patch.object(export_policy, "load_reference", side_effect=AssertionError("motion accessed")), \
             mock.patch.object(export_policy, "robot_parameters", side_effect=AssertionError("MJCF accessed")):
            metadata = export_policy.export_model(self.checkpoint, self.output, parity_samples=3)
        self.assertEqual(list(self.directory.iterdir()), [self.output])
        graph = onnx.load(self.output)
        onnx.checker.check_model(graph)
        embedded = json.loads({entry.key: entry.value for entry in graph.metadata_props}["supermimic"])
        self.assertEqual(embedded, metadata)
        self.assertEqual(metadata["schema"], "supermimic.policy")
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["source_checkpoint_sha256"], sha256_file(self.checkpoint))
        self.assertEqual(metadata["control_dt"], 0.02)
        for field, expected in {
            "history_length": 3, "goal_frames": 2, "observation_dim": 99,
            "goal_dim": 146, "action_dim": 29,
        }.items():
            self.assertEqual(metadata[field], expected)
        self.assertNotIn("motion", json.dumps(metadata).lower())
        for tensors, expected in (
            (graph.graph.input, {"obs_hist": [1, 3, 99], "goal": [1, 146]}),
            (graph.graph.output, {"action": [1, 29]}),
        ):
            self.assertEqual({tensor.name for tensor in tensors}, set(expected))
            for tensor in tensors:
                self.assertEqual(tensor.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                self.assertEqual([dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim], expected[tensor.name])
        if importlib.util.find_spec("onnxruntime"):
            self.assertTrue(metadata["verification"]["onnxruntime_checked"])
            self.assertLess(metadata["verification"]["maximum_absolute_action_error"], 1e-5)
            parity = export_policy.build_model_parity_samples(self.policy, 3)
            parity["samples"][1]["action"][0] += 1.0
            with self.assertRaisesRegex(ValueError, "parity failed"):
                export_policy.validate_onnx_parity(self.output, parity, 3)

    def test_parity_samples_use_deterministic_bounded_checkpoint_distributions(self):
        random_state = torch.random.get_rng_state().clone()
        parity = export_policy.build_model_parity_samples(self.policy, 4)
        self.assertEqual(parity, export_policy.build_model_parity_samples(self.policy, 4))
        torch.testing.assert_close(random_state, torch.random.get_rng_state())
        self.assertEqual(len(parity["samples"]), 4)
        for sample in parity["samples"]:
            for name, normalizer, shape in (
                ("history", self.policy.obs_norm, (1, 3, 99)),
                ("goal", self.policy.goal_norm, (1, 146)),
            ):
                features = torch.tensor(sample[name]).reshape(shape)
                normalized = normalizer(features)
                self.assertTrue(torch.isfinite(normalized).all())
                self.assertLessEqual(normalized.abs().max().item(), 2.00001)
                self.assertGreater(normalized.std().item(), 0.5)
                self.assertTrue(np.any(np.asarray(sample[name]) != 0))
        self.assertNotEqual(parity["samples"][0]["history"], parity["samples"][1]["history"])
        self.assertNotEqual(parity["samples"][0]["goal"], parity["samples"][1]["goal"])
        with self.assertRaisesRegex(ValueError, "at least two"):
            export_policy.build_model_parity_samples(self.policy, 1)

    def test_existing_destinations_including_symlinks_are_never_followed(self):
        sentinel = self.directory / "sentinel"
        sentinel.write_bytes(b"preserve existing output")
        for destination_kind in ("file", "directory", "symlink", "dangling_symlink"):
            destination = self.directory / f"{destination_kind}.onnx"
            if destination_kind == "file":
                destination.write_bytes(b"existing file")
            elif destination_kind == "directory":
                destination.mkdir()
            else:
                destination.symlink_to(sentinel if destination_kind == "symlink" else self.directory / "missing")
            with self.subTest(destination_kind=destination_kind), \
                 mock.patch.object(export_policy, "load_policy") as loader, \
                 self.assertRaises(FileExistsError):
                export_policy.export_model(self.checkpoint, destination)
            loader.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"preserve existing output")
        self.assertEqual((self.directory / "file.onnx").read_bytes(), b"existing file")
        self.assertFalse((self.directory / "missing").exists())

    def test_failed_verification_and_racing_publication_leave_no_partial_model(self):
        with mock.patch.object(export_policy, "validate_onnx_parity", side_effect=ValueError("verification failed")):
            with self.assertRaisesRegex(ValueError, "verification failed"):
                export_policy.export_model(self.checkpoint, self.output, parity_samples=2)
        self.assertEqual(list(self.directory.iterdir()), [])

        changing_checkpoint = self.directory / "changing.pt"
        changing_checkpoint.write_bytes(self.checkpoint.read_bytes())

        def change_checkpoint_during_verification(*arguments):
            with changing_checkpoint.open("ab") as stream:
                stream.write(b"changed during export")
            return {"onnxruntime_checked": False, "reason": "test checkpoint mutation"}

        with mock.patch.object(export_policy, "validate_onnx_parity", side_effect=change_checkpoint_during_verification):
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                export_policy.export_model(changing_checkpoint, self.output, parity_samples=2)
        changing_checkpoint.unlink()
        self.assertEqual(list(self.directory.iterdir()), [])

        def race_before_publication(*arguments):
            self.output.symlink_to(self.directory / "missing")
            return {"onnxruntime_checked": False, "reason": "test race"}

        with mock.patch.object(export_policy, "validate_onnx_parity", side_effect=race_before_publication):
            with self.assertRaises(FileExistsError):
                export_policy.export_model(self.checkpoint, self.output, parity_samples=2)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(list(self.directory.iterdir()), [self.output])

    def test_standalone_cli_reports_missing_runtime_without_sidecars(self):
        import onnx

        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"onnxruntime": None}), contextlib.redirect_stdout(output):
            status = export_policy.main([
                "--checkpoint", str(self.checkpoint), "--output", str(self.output), "--parity", "2",
            ])
        self.assertEqual(status, 0)
        self.assertIn("parity SKIPPED: onnxruntime is not installed", output.getvalue())
        graph = onnx.load(self.output)
        metadata = json.loads({entry.key: entry.value for entry in graph.metadata_props}["supermimic"])
        self.assertFalse(metadata["verification"]["onnxruntime_checked"])
        self.assertEqual(list(self.directory.iterdir()), [self.output])

    def test_cli_preserves_bundle_mode_and_rejects_partial_bundle_arguments(self):
        bundle_arguments = [
            "--checkpoint", str(self.checkpoint), "--output", str(self.directory / "bundle"),
            "--motion", "motion.csv", "--mjcf", "robot.xml", "--parity", "1",
        ]
        with mock.patch.object(export_policy, "export_bundle", return_value={"frame_count": 17}) as exporter, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(export_policy.main(bundle_arguments), 0)
        self.assertIn("Exported 17 frames", output.getvalue())
        exporter.assert_called_once_with(
            checkpoint=self.checkpoint, output=self.directory / "bundle", motion=Path("motion.csv"),
            mjcf=Path("robot.xml"), src_fps=30.0, parity_samples=1,
        )
        for argument in ("--motion", "--mjcf"):
            with self.subTest(argument=argument), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as failure:
                export_policy.main(["--checkpoint", str(self.checkpoint), "--output", str(self.output), argument, "input"])
            self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
