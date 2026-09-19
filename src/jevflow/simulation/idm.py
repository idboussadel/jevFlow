"""Vectorised Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]

ACCEL_EXPONENT = 4.0
MAX_PHYSICAL_DECEL = 9.0  # m/s², emergency braking on dry asphalt
MIN_GAP_EPS = 0.1


def idm_acceleration(v: F, v0: F, gap: F, dv: F, T: F, a: F, b: F, s0: F) -> F:
    """IDM acceleration.

    ``gap`` is the bumper-to-bumper distance to the leader (``inf`` for free road) and
    ``dv`` the approach rate ``v - v_leader``.
    """
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a * b)))
    interaction = (s_star / np.maximum(gap, MIN_GAP_EPS)) ** 2
    free = (v / np.maximum(v0, 0.1)) ** ACCEL_EXPONENT
    acc = a * (1.0 - free - interaction)
    return np.clip(acc, -MAX_PHYSICAL_DECEL, a)


def ballistic_update(s: F, v: F, acc: F, dt: float) -> tuple[F, F]:
    """Ballistic position update that never lets a vehicle roll backwards."""
    v_next = v + acc * dt
    stopping = v_next < 0.0
    safe_acc = np.where(stopping, np.minimum(acc, -1e-9), -1.0)  # both branches are evaluated
    ds = np.where(stopping, -0.5 * v * v / safe_acc, v * dt + 0.5 * acc * dt * dt)
    return s + ds, np.maximum(v_next, 0.0)
