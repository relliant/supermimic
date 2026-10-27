"""Turn timestamped Pico qpos36 into measured, uniform full-state references.

Only NumPy is required. Input quaternions are wxyz; output columns are root
xyz, quaternion wxyz, joints29, joint velocities29, world linear velocity3,
and world angular velocity3. No pose or velocity is extrapolated into the
future. The SDK adapter owns clock mapping and must clear/rearm after a source
sequence or timestamp discontinuity.
"""

from __future__ import annotations

from collections import deque
import math
from numbers import Integral, Real

import numpy as np

CONTROL_DT = 0.02


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(quaternion), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("root quaternion must have nonzero norm")
    scaled = quaternion / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True,
    )
    vector = (
        left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar, vector), axis=-1)


def _slerp(start: np.ndarray, end: np.ndarray, fraction: np.ndarray | float) -> np.ndarray:
    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0, -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    tangent = end - dot * start
    tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    angle = np.arctan2(tangent_norm, dot)
    weight = np.asarray(fraction)[..., None]
    tangent_weight = np.divide(
        np.sin(weight * angle), tangent_norm,
        out=np.broadcast_to(weight, tangent_norm.shape).copy(),
        where=tangent_norm > 1e-12,
    )
    interpolated = np.cos(weight * angle) * start + tangent_weight * tangent
    interpolated = np.where(weight == 0, start, np.where(weight == 1, end, interpolated))
    return _normalize_quaternion(interpolated)


def _world_angular_velocity(
    start: np.ndarray, end: np.ndarray, duration: np.ndarray,
) -> np.ndarray:
    conjugate = start.copy()
    conjugate[..., 1:] *= -1
    relative = _multiply_quaternions(end, conjugate)
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    vector_norm = np.linalg.norm(relative[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(vector_norm, np.clip(relative[..., :1], 0, 1))
    logarithm_scale = np.divide(
        angle, vector_norm, out=np.full_like(vector_norm, 2.0), where=vector_norm > 1e-12,
    )
    return relative[..., 1:] * logarithm_scale / duration[:, None]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


class BufferedReference:
    """A bounded, session-local buffer with a fixed 20 ms source timeline.

    ``chunk()`` returns ``(first_row_unix_time, rows71)``. The first chunk starts
    near newest - (K * 20 ms + jitter_margin + 20 ms), not at the beginning of
    a long capture. Later chunks preserve overlap back to that first chunk,
    rolling forward only when their span exceeds ``history_seconds``. A
    receiver can therefore keep its own delayed cursor while merging chunks.

    Filtering is a causal exponential low-pass on accepted qpos, using actual
    source dt and quaternion SLERP. ``filter_tau=0`` disables it. Velocities are
    differentiated after uniform resampling, centered except at the session's
    very first sample. Every emitted row has a measured next grid sample.
    """

    def __init__(
        self, goal_frames: int, *, filter_tau: float = 0.03,
        jitter_margin: float = 0.04, max_gap: float = 0.15,
        history_seconds: float = 1.5, recenter: bool = True,
    ) -> None:
        if isinstance(goal_frames, (bool, np.bool_)) or not isinstance(goal_frames, Integral) or goal_frames < 1:
            raise ValueError("goal_frames must be a positive integer")
        self.goal_frames = int(goal_frames)
        self.filter_tau = _finite_number(filter_tau, "filter_tau")
        self.jitter_margin = _finite_number(jitter_margin, "jitter_margin")
        self.max_gap = _finite_number(max_gap, "max_gap")
        self.history_seconds = _finite_number(history_seconds, "history_seconds")
        if self.filter_tau < 0 or self.jitter_margin < 0 or self.max_gap <= 0:
            raise ValueError("filter_tau and jitter_margin must be nonnegative; max_gap must be positive")
        self.startup_delay_seconds = self.goal_frames * CONTROL_DT + self.jitter_margin + CONTROL_DT
        if self.history_seconds < self.startup_delay_seconds:
            raise ValueError("history_seconds must cover the startup delay and future horizon")
        self.recenter = bool(recenter)
        self.clear()

    def clear(self) -> None:
        """Forget the timeline, filter, recenter transform, and source metadata."""
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._origin_time: float | None = None
        self._source_sequence: int | None = None
        self._source_time: float | None = None
        self._first_emitted_index: int | None = None
        self._reference_time: float | None = None
        self._delay_seconds: float | None = None
        self._last_row_count = 0
        self._accepted_samples = 0
        self._duplicates = 0
        self._initial_xy = np.zeros(2)
        self._recenter_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def source_sequence(self) -> int | None:
        return self._source_sequence

    @property
    def source_time(self) -> float | None:
        return self._source_time

    @property
    def delay_seconds(self) -> float | None:
        """Newest accepted time minus the first row of the last returned chunk."""
        return self._delay_seconds

    @property
    def diagnostics(self) -> dict[str, int | float | None]:
        raw_duration = self._samples[-1][0] - self._samples[0][0] if self._samples else 0.0
        return {
            "source_sequence": self.source_sequence, "source_time": self.source_time,
            "raw_samples": len(self._samples), "raw_duration_seconds": raw_duration,
            "accepted_samples": self._accepted_samples, "duplicates": self._duplicates,
            "reference_time": self._reference_time, "emitted_rows": self._last_row_count,
            "delay_seconds": self.delay_seconds, "startup_delay_seconds": self.startup_delay_seconds,
            "history_seconds": self.history_seconds,
            "raw_retention_limit_seconds": self.history_seconds + 3 * CONTROL_DT + self.max_gap,
        }

    def add(self, qpos: np.ndarray, *, source_sequence: int, source_time: float) -> bool:
        """Accept one qpos36, ignore exact duplicates, reject discontinuities.

        Errors leave accepted state unchanged; callers must explicitly clear
        and rearm, rather than silently bridging a tracking outage.
        """
        if isinstance(source_sequence, (bool, np.bool_)) or not isinstance(source_sequence, Integral) or source_sequence < 0:
            raise ValueError("source_sequence must be a nonnegative integer")
        timestamp = _finite_number(source_time, "source_time")
        if self._source_time is not None:
            if source_sequence == self._source_sequence and timestamp == self._source_time:
                self._duplicates += 1
                return False
            if source_sequence <= self._source_sequence or timestamp <= self._source_time:
                raise ValueError("source sequence/time regression; clear and rearm")
            source_dt = timestamp - self._source_time
            timestamp_roundoff = 4 * abs(np.spacing(timestamp))
            if source_dt > self.max_gap + timestamp_roundoff:
                raise ValueError("source time gap exceeds max_gap; clear and rearm")
        else:
            source_dt = 0.0
        try:
            position = np.asarray(qpos)
            if position.shape != (36,) or position.dtype.kind not in "fiu":
                raise ValueError("qpos must have shape (36,) with numeric values")
            position = position.astype(np.float64, copy=True)
        except (TypeError, OverflowError) as error:
            raise ValueError("qpos must have shape (36,) with numeric values") from error
        if not np.isfinite(position).all():
            raise ValueError("qpos must contain only finite values")
        position[3:7] = _normalize_quaternion(position[3:7])

        if self._origin_time is None:
            self._origin_time = timestamp
            if self.recenter:
                self._initial_xy = position[:2].copy()
                scalar, axis_x, axis_y, axis_z = position[3:7]
                yaw = math.atan2(2 * (scalar * axis_z + axis_x * axis_y), 1 - 2 * (axis_y**2 + axis_z**2))
                self._recenter_quaternion = np.array([math.cos(yaw / 2), 0.0, 0.0, -math.sin(yaw / 2)])
        if self.recenter:
            position[:2] -= self._initial_xy
            yaw_cosine = 1 - 2 * self._recenter_quaternion[3] ** 2
            yaw_sine = 2 * self._recenter_quaternion[0] * self._recenter_quaternion[3]
            root_x, root_y = position[:2]
            position[:2] = (yaw_cosine * root_x - yaw_sine * root_y, yaw_sine * root_x + yaw_cosine * root_y)
            position[3:7] = _multiply_quaternions(self._recenter_quaternion, position[3:7])
        if self._samples:
            previous = self._samples[-1][1]
            if np.dot(position[3:7], previous[3:7]) < 0:
                position[3:7] *= -1
            if self.filter_tau > 0:
                fraction = -math.expm1(-source_dt / self.filter_tau)
                quaternion = _slerp(previous[3:7], position[3:7], fraction)
                position = previous * (1 - fraction) + position * fraction
                position[3:7] = quaternion
        elapsed = timestamp - self._origin_time
        self._samples.append((elapsed, position))
        self._source_time = timestamp
        self._source_sequence = int(source_sequence)
        self._accepted_samples += 1
        # Three grid steps cover the derivative guards and off-grid arrival.
        # Retain one preceding source sample as the interpolation bracket.
        cutoff = elapsed - self.history_seconds - 3 * CONTROL_DT
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        return True

    def chunk(self) -> tuple[float, np.ndarray] | None:
        """Return overlapping measured history on the fixed grid, or await startup."""
        if len(self._samples) < 2:
            return None
        elapsed = self._samples[-1][0]
        # Unix float timestamps lose sub-microsecond precision. Work in elapsed
        # time, allowing only their roundoff when recognizing grid boundaries.
        tolerance = 4 * abs(np.spacing(self._source_time))
        if elapsed + tolerance < self.startup_delay_seconds:
            return None
        newest_grid_index = math.floor((elapsed + tolerance) / CONTROL_DT)
        last_index = newest_grid_index - 1
        if self._first_emitted_index is None:
            self._first_emitted_index = max(0, math.floor(
                (elapsed - self.startup_delay_seconds + tolerance) / CONTROL_DT,
            ))
        history_frames = math.floor(self.history_seconds / CONTROL_DT)
        first_index = max(self._first_emitted_index, last_index - history_frames)
        guard_index = max(0, first_index - 1)
        grid_indices = np.arange(guard_index, last_index + 2)
        grid_times = grid_indices * CONTROL_DT
        raw_times = np.array([sample[0] for sample in self._samples])
        raw_positions = np.stack([sample[1] for sample in self._samples])
        if grid_times[0] < raw_times[0] - tolerance or grid_times[-1] > raw_times[-1] + tolerance:
            raise ValueError("insufficient measured samples for reference derivatives")
        # Clamp floating point boundary noise only, never predict a future pose.
        interpolation_times = np.clip(grid_times, raw_times[0], raw_times[-1])
        brackets = np.searchsorted(raw_times, interpolation_times, side="right") - 1
        brackets = np.clip(brackets, 0, len(raw_times) - 2)
        # An exactly grid-aligned Unix timestamp may differ from its elapsed
        # grid time by an ULP. Always select that measured endpoint so later
        # arrivals cannot revise already emitted derivatives through roundoff.
        for endpoint_indices in (brackets, brackets + 1):
            endpoint_times = raw_times[endpoint_indices]
            interpolation_times = np.where(
                np.abs(grid_times - endpoint_times) <= tolerance,
                endpoint_times, interpolation_times,
            )
        fractions = (interpolation_times - raw_times[brackets]) / (raw_times[brackets + 1] - raw_times[brackets])
        before = raw_positions[brackets]
        after = raw_positions[brackets + 1]
        uniform = before * (1 - fractions[:, None]) + after * fractions[:, None]
        uniform[:, 3:7] = _slerp(before[:, 3:7], after[:, 3:7], fractions)

        selected = np.arange(first_index - guard_index, last_index - guard_index + 1)
        previous_indices = np.maximum(selected - 1, 0)
        next_indices = selected + 1
        durations = (next_indices - previous_indices) * CONTROL_DT
        differences = (uniform[next_indices] - uniform[previous_indices]) / durations[:, None]
        angular_velocity = _world_angular_velocity(
            uniform[previous_indices, 3:7], uniform[next_indices, 3:7], durations,
        )
        rows = np.concatenate((
            uniform[selected], differences[:, 7:36], differences[:, :3], angular_velocity,
        ), axis=1)
        if not np.isfinite(rows).all():
            raise ValueError("reference interpolation produced nonfinite full-state rows")
        reference_time = self._origin_time + first_index * CONTROL_DT
        self._reference_time = reference_time
        self._delay_seconds = self._source_time - reference_time
        self._last_row_count = len(rows)
        return reference_time, rows
