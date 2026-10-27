"""Opt-in actual ONNX/Redis integration, using recorded poses, not Pico hardware.

SUPERMIMIC_ONNX=... .venv/bin/python -m unittest deployment.tests.test_live_integration -v
Uses a temporary loopback Redis port and removes every temporary output/server.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

import numpy as np


@unittest.skipUnless(os.environ.get('SUPERMIMIC_ONNX'), 'opt-in learned ONNX path is required')
class LiveIntegrationTests(unittest.TestCase):
    def test_buffered_pose_replay_and_bounded_underrun(self):
        import redis
        from deployment.common.contract import load_reference
        from deployment.streaming.live import LivePublisher, LiveReferencePacket
        from deployment.teleop.reference import BufferedReference

        root = Path(__file__).resolve().parents[2]
        policy = Path(os.environ['SUPERMIMIC_ONNX']).resolve(strict=True)
        server_executable = shutil.which('redis-server')
        self.assertIsNotNone(server_executable)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        reference = load_reference(root / 'data/lafan1_g1/walk1_subject1.csv')
        with tempfile.TemporaryDirectory(prefix='supermimic-live-validation-') as directory:
            temporary = Path(directory)
            server = subprocess.Popen([
                server_executable, '--bind', '127.0.0.1', '--port', str(port),
                '--save', '', '--appendonly', 'no',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            client = redis.Redis(port=port, socket_timeout=.5)
            controller = publisher = None
            try:
                for _ in range(100):
                    try:
                        if client.ping():
                            break
                    except redis.exceptions.ConnectionError:
                        time.sleep(.02)
                with (temporary / 'controller.log').open('w+') as log:
                    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    controller = subprocess.Popen([
                        sys.executable, '-m', 'deployment.sim2sim', '--policy', str(policy),
                        '--listen', '--headless', '--stream-mode', 'live', '--redis-port', str(port),
                        '--channel', 'supermimic:live-test', '--hold-timeout', '.12',
                        '--metrics', str(temporary / 'metrics.json'), '--record', str(temporary / 'states.npz'),
                    ], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    publisher = LivePublisher(client, 'supermimic:live-test', timeout=15)
                    publisher.acquire()
                    status = publisher.wait_ready()
                    self.assertEqual(status['state'], 'waiting')
                    buffer = BufferedReference(status['goal_frames'], filter_tau=0, recenter=False)
                    first_monotonic, first_unix = time.monotonic(), time.time()
                    session = uuid.uuid4().hex
                    sequence = 0
                    packet = None
                    for sample_index in range(75):
                        deadline = first_monotonic + sample_index * .02
                        time.sleep(max(0, deadline - time.monotonic()))
                        buffer.add(reference[sample_index, :36], source_sequence=sample_index,
                                   source_time=first_unix + sample_index * .02)
                        chunk = buffer.chunk()
                        if chunk is None:
                            continue
                        reference_time, rows = chunk
                        packet = LiveReferencePacket(
                            status['controller_id'], session, sequence, time.time(),
                            'start' if sequence == 0 else 'frame', 'offline-live-replay',
                            buffer.source_sequence, buffer.source_time, reference_time, rows,
                        )
                        if sequence == 0:
                            publisher.start(packet)
                        else:
                            publisher.publish(packet)
                        sequence += 1
                    time.sleep(.3)
                    status = publisher.read_status()
                    self.assertEqual(status['state'], 'stale', status)
                    self.assertGreater(status['control_steps'], 40)
                    self.assertGreater(status['held_steps'], 0)
                    self.assertGreater(status['reference_delay_seconds'], .2)
                    frozen_steps = status['control_steps']
                    time.sleep(.08)
                    self.assertEqual(publisher.read_status()['control_steps'], frozen_steps)
                    # Replaying an old session cannot restart physics.
                    publisher.publish(replace(packet, sequence=sequence, sent_at=time.time()))
                    time.sleep(.04)
                    self.assertEqual(publisher.read_status()['control_steps'], frozen_steps)
                    controller.send_signal(signal.SIGINT)
                    self.assertEqual(controller.wait(timeout=10), 0)
                    report = json.loads((temporary / 'metrics.json').read_text())
                    self.assertEqual(report['total_control_steps'], frozen_steps)
                    with np.load(temporary / 'states.npz', allow_pickle=False) as recording:
                        self.assertTrue(np.isfinite(recording['joint_position']).all())
                        self.assertEqual(len(recording['holding']), frozen_steps)
                        self.assertTrue(np.any(recording['holding']))
                    print('LIVE_REPLAY_VALIDATED', json.dumps(status), flush=True)
            finally:
                if publisher is not None:
                    publisher.close()
                if controller is not None and controller.poll() is None:
                    controller.send_signal(signal.SIGINT)
                    try:
                        controller.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        controller.kill()
                        controller.wait()
                if (temporary / 'controller.log').exists():
                    print((temporary / 'controller.log').read_text(), flush=True)
                client.close()
                server.terminate()
                server.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
