"""Shared test helpers."""

from __future__ import annotations

from jevflow.domain.enums import ControlAction, Interval
from jevflow.sensing.aggregator import TrafficStateAggregator
from jevflow.sensing.detectors import DetectorArray
from jevflow.simulation.engine import SimulationEngine


def run_fixed_cycle(
    engine: SimulationEngine,
    until: float,
    green: float = 30.0,
    aggregator: TrafficStateAggregator | None = None,
    detectors: DetectorArray | None = None,
) -> None:
    """Drive the engine with a simple fixed cycle (tests only)."""
    while engine.time < until:
        signal = engine.signal
        if signal.interval is Interval.GREEN and signal.green_elapsed >= green and signal.can_terminate():
            signal.command(ControlAction.SWITCH_PHASE)
        engine.step(True)
        if detectors is not None:
            detectors.sample()
        if aggregator is not None:
            aggregator.update()
