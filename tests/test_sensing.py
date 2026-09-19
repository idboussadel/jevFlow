from __future__ import annotations

import numpy as np

from jevflow.domain.enums import Approach
from jevflow.sensing.aggregator import TrafficStateAggregator
from jevflow.sensing.detectors import DetectorArray
from jevflow.simulation.engine import SimulationEngine
from jevflow.simulation.scenario import FaultEvent, SensorProfile
from jevflow.simulation.scenarios import get_scenario
from tests.helpers import run_fixed_cycle


def _setup(
    key: str, seed: int = 5, sensors: SensorProfile | None = None
) -> tuple[SimulationEngine, DetectorArray, TrafficStateAggregator]:
    scenario = get_scenario(key)
    if sensors is not None:
        scenario = scenario.model_copy(update={"sensors": sensors})
    engine = SimulationEngine(scenario, seed)
    detectors = DetectorArray(engine, scenario.sensors)
    return engine, detectors, TrafficStateAggregator(engine, detectors)


def test_stop_line_counts_match_departures_within_miss_rate() -> None:
    engine, detectors, aggregator = _setup(
        "balanced_midday", sensors=SensorProfile(miss_rate=0.02, false_actuations_per_hour=0)
    )
    run_fixed_cycle(engine, 600, detectors=detectors, aggregator=aggregator)
    counted = sum(detectors.get(a, lane, "stop").rising_edges for a in Approach for lane in (0, 1))
    departed = int(engine.departures.sum())
    assert 0.9 * departed <= counted <= 1.02 * departed


def test_controller_never_sees_ground_truth_but_estimates_track_it() -> None:
    engine, detectors, aggregator = _setup("balanced_midday")
    errors, truths = [], []
    while engine.time < 900:
        run_fixed_cycle(engine, engine.time + 1.0, detectors=detectors, aggregator=aggregator)
        est, tru = aggregator.estimated_queues(), engine.true_queues()
        errors += [abs(est[a] - tru[a]) for a in Approach]
        truths += [tru[a] for a in Approach]
    assert np.mean(errors) < max(2.5, 0.6 * np.mean(truths))


def test_stuck_on_loop_is_flagged_and_reported_as_degraded() -> None:
    fault = FaultEvent(
        approach=Approach.NORTH, lane=1, detector="stop", mode="stuck_on", start_s=30, duration_s=600
    )
    engine, detectors, aggregator = _setup("balanced_midday", sensors=SensorProfile(faults=(fault,)))
    run_fixed_cycle(engine, 200, detectors=detectors, aggregator=aggregator)
    obs = aggregator.observe()
    assert "north.1.stop" in obs.approaches[Approach.NORTH].faulty_detectors
    assert obs.approaches[Approach.NORTH].detector_health.value == "degraded"
    assert obs.approaches[Approach.SOUTH].detector_health.value == "ok"


def test_observation_is_valid_pydantic_and_bounded() -> None:
    engine, detectors, aggregator = _setup("rush_hour")
    run_fixed_cycle(engine, 240, detectors=detectors, aggregator=aggregator)
    obs = aggregator.observe()
    for m in obs.approaches.values():
        assert 0 <= m.occupancy_percent <= 100
        assert m.average_speed_kmh >= 0
    assert obs.model_dump_json()
