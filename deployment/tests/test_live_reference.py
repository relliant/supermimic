"""Live references use acquisition time and control ticks, not packet count."""

from dataclasses import replace

import numpy as np
import pytest

from deployment.live_reference import LiveReferencePacket, LiveReferenceReceiver, decode_live
from deployment.reference_stream import decode


class FeedbackSimulation:
    def __init__(self):
        self.resets = 0
        self.steps = 0
        self.history = []
        self.references = []

    def accept_reference(self, reference, name, *, reset):
        if reset:
            self.resets += 1
            self.history = []
        self.reference = reference.copy()

    def advance(self):
        self.steps += 1
        self.history.append(self.steps)
        self.references.append(self.reference.copy())


def make_packet(*, frame_count=16, now=100.0):
    rows = np.zeros((frame_count, 71), dtype=np.float32)
    rows[:, 3] = 1
    rows[:, 0] = np.arange(frame_count) * 0.02
    return LiveReferencePacket('controller', 'session', 0, now, 'start', 'pico',
                               1000, now, now - 0.02 * (frame_count - 1), rows)


def test_codec_separates_all_clocks_and_preserves_v1_boundary():
    packet = make_packet()
    restored = decode_live(packet.encode(), 10, now=100.01, max_age=0.5)
    assert restored.source_time == 100
    assert restored.reference_time == pytest.approx(99.7)
    np.testing.assert_array_equal(restored.reference, packet.reference)
    with pytest.raises(ValueError):
        decode(packet.encode(), 10, now=100.01, max_age=0.5)
    with pytest.raises(ValueError, match='stale'):
        decode_live(replace(packet, sent_at=101).encode(), 10, now=101, max_age=0.5)
    with pytest.raises(ValueError, match='unobserved future'):
        replace(packet, reference_time=100).encode()
    with pytest.raises(ValueError, match=r'K\+1'):
        decode_live(replace(packet, reference=packet.reference[:5]).encode(), 10, now=100, max_age=0.5)


def test_ingestion_does_not_step_and_control_continues_without_new_packets():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    packet = make_packet()
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    assert simulation.resets == 1
    assert simulation.steps == 0
    assert receiver.state == 'buffering'
    assert not receiver.tick(simulation, wall_time=100, monotonic_time=0)
    for index in range(6):
        assert receiver.tick(simulation, wall_time=100 + index * .02, monotonic_time=index * .02)
    assert simulation.steps == 6
    assert receiver.cursor == 6
    # No new complete future window: retain the last window, NOT an action.
    assert receiver.tick(simulation, wall_time=100.12, monotonic_time=.12)
    assert receiver.state == 'holding'
    assert receiver.reference_time == pytest.approx(packet.reference_time + 5 * .02)
    assert receiver.cursor == 6
    assert receiver.tick(simulation, wall_time=100.14, monotonic_time=.14)
    assert len(simulation.history) == 8
    np.testing.assert_array_equal(simulation.references[-1], simulation.references[-2])
    assert not receiver.tick(simulation, wall_time=100.26, monotonic_time=.26)
    assert receiver.state == 'stale'
    assert simulation.steps == 8


@pytest.mark.parametrize('arrival_stride', [1, 2, 3])
def test_overlapping_chunks_merge_without_clocking_control_on_arrival(arrival_stride):
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    packet = make_packet(frame_count=30)
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    receiver.tick(simulation, wall_time=100, monotonic_time=0)
    for tick in range(15):
        now = 100 + (tick + 1) * .02
        if tick % arrival_stride == 0:
            updated = replace(packet, state='frame', sequence=10 + tick, source_sequence=1001 + tick,
                              source_time=now, sent_at=now)
            receiver.ingest(updated.encode(), simulation, wall_time=now)
            receiver.ingest(updated.encode(), simulation, wall_time=now)  # duplicate has no extra tick
        assert receiver.tick(simulation, wall_time=now, monotonic_time=tick * .02)
    assert simulation.steps == 15
    assert simulation.resets == 1
    assert receiver.cursor == 15


def test_same_sdk_sample_cannot_be_refreshed_by_new_publication_time():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    packet = make_packet()
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    invalid = replace(packet, state='frame', sequence=1, source_time=100.1, sent_at=100.1)
    receiver.ingest(invalid.encode(), simulation, wall_time=100.1)
    assert receiver.state == 'stale'
    assert simulation.steps == 0
    receiver.ingest(replace(invalid, sequence=2).encode(), simulation, wall_time=100.1)
    assert receiver.state == 'stale'


def test_late_source_freezes_even_with_fresh_publication_and_complete_cache():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    packet = make_packet(frame_count=30)
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    receiver.tick(simulation, wall_time=100, monotonic_time=0)
    assert not receiver.tick(simulation, wall_time=100.51, monotonic_time=.51)
    assert receiver.state == 'stale'
    assert simulation.steps == 0
    restart = replace(packet, session_id='fresh', source_sequence=2000, source_time=100.6,
                      reference_time=100.02, sent_at=100.6)
    receiver.ingest(restart.encode(), simulation, wall_time=100.6)
    assert receiver.state == 'buffering'
    assert simulation.resets == 2


def test_expired_foreign_and_consumed_packets_do_not_disarm_active_session():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    packet = make_packet()
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    foreign = replace(make_packet(now=90), controller_id='previous-controller')
    receiver.ingest(foreign.encode(), simulation, wall_time=100.1)
    assert receiver.state == 'buffering'
    receiver.ingest(packet.encode(), simulation, wall_time=100.6)
    assert receiver.state == 'buffering'  # consumed packet ignored; tick owns the source watchdog
    assert not receiver.tick(simulation, wall_time=100.6, monotonic_time=0.6)
    assert receiver.state == 'stale'


def test_wall_clock_adjustment_cannot_extend_monotonic_source_watchdog():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    receiver.ingest(make_packet().encode(), simulation, wall_time=100)
    receiver.tick(simulation, wall_time=100, monotonic_time=0)
    assert not receiver.tick(simulation, wall_time=99.99, monotonic_time=.6)
    assert receiver.state == 'stale'


def test_pause_stop_rejects_cached_session_and_reset_failure_is_contained():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10)
    packet = make_packet()
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    receiver.stop('viewer paused')
    receiver.ingest(replace(packet, state='frame', sequence=1).encode(), simulation, wall_time=100)
    assert not receiver.tick(simulation, wall_time=100, monotonic_time=0)
    assert simulation.steps == 0
    def failed_reset(*arguments, **keywords):
        raise ValueError('bad reset')
    simulation.accept_reference = failed_reset
    receiver.ingest(replace(packet, session_id='other').encode(), simulation, wall_time=100)
    assert receiver.state == 'stale'


def test_discontinuity_and_excessive_lag_require_rearm():
    simulation = FeedbackSimulation()
    receiver = LiveReferenceReceiver('controller', 10, max_delay=.25)
    packet = make_packet()
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    assert not receiver.tick(simulation, wall_time=100, monotonic_time=0)
    assert receiver.state == 'stale'
    receiver = LiveReferenceReceiver('controller', 10)
    receiver.ingest(packet.encode(), simulation, wall_time=100)
    gap = replace(packet, state='frame', sequence=3, source_sequence=1002,
                  sent_at=100.4, source_time=100.4, reference_time=100.02)
    receiver.ingest(gap.encode(), simulation, wall_time=100.4)
    assert receiver.state == 'stale'
