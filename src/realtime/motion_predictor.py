"""Motion prediction for smooth real-time display.

When no new inference data is available, extrapolates the last known
pose using velocity estimation with exponential decay. This eliminates
visible freezes between GVHMR inference windows.
"""

import time

import numpy as np


class MotionPredictor:
    """Predicts forward motion to fill gaps between inference windows.

    Uses exponential moving average of velocity to extrapolate poses.
    Velocity decays exponentially over time to prevent unbounded drift.

    The displacement integral of v(t) = v0 * exp(-k*t) over [0, dt] is:
        dx = v0/k * (1 - exp(-k*dt))
    This naturally bounds the total extrapolation to v0/k, preventing
    runaway predictions when inference is delayed.
    """

    def __init__(
        self,
        velocity_alpha: float = 0.6,
        max_predict_dt: float = 0.5,
        decay_rate: float = 5.0,
    ):
        """
        Args:
            velocity_alpha: EMA weight for velocity updates (higher = more responsive).
            max_predict_dt: Maximum seconds to extrapolate ahead.
            decay_rate: Exponential decay rate (higher = prediction decays faster).
                        At decay_rate=5, velocity halves every ~0.14s.
        """
        self._alpha = velocity_alpha
        self._max_dt = max_predict_dt
        self._decay_rate = decay_rate

        self._last_qpos = None
        self._last_time = None
        self._velocity = None

    def update(self, qpos: np.ndarray):
        """Record a new observed pose and update velocity estimate."""
        now = time.monotonic()

        if self._last_qpos is not None and self._last_time is not None:
            dt = now - self._last_time
            if dt > 0.001:
                instant_vel = (qpos - self._last_qpos) / dt
                if self._velocity is None:
                    self._velocity = instant_vel.copy()
                else:
                    self._velocity = (
                        self._alpha * instant_vel
                        + (1 - self._alpha) * self._velocity
                    )

        self._last_qpos = qpos.copy()
        self._last_time = now

    def predict(self) -> np.ndarray | None:
        """Predict current pose by extrapolating from last observation.

        Returns None if no observations have been recorded.
        """
        if self._last_qpos is None:
            return None

        if self._velocity is None or self._last_time is None:
            return self._last_qpos.copy()

        dt = time.monotonic() - self._last_time
        if dt < 0.001:
            return self._last_qpos.copy()

        dt = min(dt, self._max_dt)

        # Integral of v * exp(-k*t) from 0 to dt = v/k * (1 - exp(-k*dt))
        # This bounds total displacement to v/k as dt → ∞
        k = self._decay_rate
        if k > 0:
            effective_dt = (1.0 - np.exp(-k * dt)) / k
        else:
            effective_dt = dt

        predicted = self._last_qpos + self._velocity * effective_dt

        # Re-normalize root quaternion (qpos indices 3:7)
        quat = predicted[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            predicted[3:7] = quat / quat_norm

        return predicted

    @property
    def has_velocity(self) -> bool:
        """Whether we have enough history to extrapolate."""
        return self._velocity is not None

    @property
    def time_since_update(self) -> float:
        """Seconds since last update, or inf if never updated."""
        if self._last_time is None:
            return float("inf")
        return time.monotonic() - self._last_time
