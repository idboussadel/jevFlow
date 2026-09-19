from __future__ import annotations

import pytest

from jevflow.control.base import DecisionContext
from jevflow.control.safety import legal_actions
from jevflow.sensing.aggregator import TrafficStateAggregator
from jevflow.sensing.detectors import DetectorArray
from jevflow.simulation.engine import SimulationEngine
from jevflow.simulation.scenarios import get_scenario


@pytest.fixture
def rush_engine() -> SimulationEngine:
    return SimulationEngine(get_scenario("rush_hour"), seed=7)


@pytest.fixture
def decision_context() -> DecisionContext:
    """A realistic context taken from a running simulation at a decision point."""
    engine = SimulationEngine(get_scenario("rush_hour"), seed=3)
    detectors = DetectorArray(engine, engine.scenario.sensors)
    aggregator = TrafficStateAggregator(engine, detectors)
    while not (engine.signal.can_terminate() and engine.time > 60):
        engine.step(True)
        detectors.sample()
        aggregator.update()
    return DecisionContext(aggregator.observe(), legal_actions(engine.signal), engine.timing)
