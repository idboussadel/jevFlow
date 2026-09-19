from __future__ import annotations

import numpy as np

from jevflow.domain.enums import Approach, ControlAction
from jevflow.simulation.engine import SimulationEngine
from jevflow.simulation.scenarios import get_scenario
from tests.helpers import run_fixed_cycle


def test_same_seed_is_reproducible() -> None:
    a = SimulationEngine(get_scenario("balanced_midday"), seed=11)
    b = SimulationEngine(get_scenario("balanced_midday"), seed=11)
    run_fixed_cycle(a, 200)
    run_fixed_cycle(b, 200)
    assert [t.delay for t in a.trips] == [t.delay for t in b.trips]


def test_vehicles_are_conserved(rush_engine: SimulationEngine) -> None:
    scheduled = len(rush_engine._pending)
    run_fixed_cycle(rush_engine, 300)
    released = scheduled - len(rush_engine._pending)
    spawned = rush_engine.fleet._next_vid - 1
    assert released == spawned + sum(rush_engine.vertical_queue().values())
    removed = spawned - len(rush_engine.fleet)
    assert removed <= len(rush_engine.trips) <= spawned


def test_saturation_flow_is_realistic(rush_engine: SimulationEngine) -> None:
    """Queue discharge on a through lane should be ~1,500–2,000 veh/h (HCM ideal 1,900)."""
    times: list[float] = []
    last = rush_engine.departures.copy()
    while rush_engine.time < 600:
        signal = rush_engine.signal
        if signal.green_elapsed >= 35 and signal.can_terminate():
            signal.command(ControlAction.SWITCH_PHASE)
        rush_engine.step(True)
        if (
            rush_engine.departures[Approach.NORTH.ordinal] > last[Approach.NORTH.ordinal]
            and 6 < signal.green_elapsed < 30
        ):
            times.append(rush_engine.time)
        last = rush_engine.departures.copy()
    headways = np.diff(times)
    headways = headways[headways < 3.5]
    flow = 3600 / float(np.median(headways))
    assert 1400 < flow < 2200


def test_red_light_violations_are_rare(rush_engine: SimulationEngine) -> None:
    run_fixed_cycle(rush_engine, 900)
    assert rush_engine.red_light_violations <= 0.01 * max(len(rush_engine.trips), 1) + 2


def test_poses_are_finite_and_inside_the_map(rush_engine: SimulationEngine) -> None:
    run_fixed_cycle(rush_engine, 120)
    poses = rush_engine.poses()
    limit = rush_engine.geometry.approach_length_m + rush_engine.geometry.stop_line_offset_m + 1
    assert np.all(np.isfinite(poses.x)) and np.all(np.isfinite(poses.heading))
    assert np.all(np.abs(poses.x) <= limit) and np.all(np.abs(poses.y) <= limit)
