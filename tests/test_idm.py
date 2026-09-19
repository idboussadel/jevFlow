from __future__ import annotations

import numpy as np

from jevflow.simulation.idm import ballistic_update, idm_acceleration


def arr(x: float) -> np.ndarray:
    return np.array([x], dtype=float)


def _one(v: float, gap: float, dv: float) -> float:
    return float(
        idm_acceleration(arr(v), arr(14.0), arr(gap), arr(dv), arr(1.2), arr(1.5), arr(2.0), arr(2.0))[0]
    )


def test_free_road_accelerates_until_desired_speed() -> None:
    assert _one(0.0, np.inf, 0.0) > 1.4
    assert abs(_one(14.0, np.inf, 0.0)) < 1e-9


def test_brakes_hard_when_closing_on_a_stopped_leader() -> None:
    assert _one(14.0, 20.0, 14.0) < -4.0


def test_vehicle_comes_to_rest_behind_obstacle_without_collision() -> None:
    s, v = np.array([0.0]), np.array([14.0])
    obstacle = 150.0
    for _ in range(600):
        acc = idm_acceleration(
            v,
            np.array([14.0]),
            obstacle - s,
            v,
            np.array([1.2]),
            np.array([1.5]),
            np.array([2.0]),
            np.array([2.0]),
        )
        s, v = ballistic_update(s, v, acc, 0.1)
    assert v[0] < 0.05
    assert 0.5 < obstacle - s[0] < 3.0


def test_ballistic_update_never_reverses() -> None:
    s, v = ballistic_update(np.array([10.0]), np.array([0.5]), np.array([-9.0]), 0.1)
    assert v[0] == 0.0 and s[0] >= 10.0
