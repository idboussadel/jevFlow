"""Industry-standard baseline controllers. Each sees only detector data, never ground truth.

* :class:`FixedTimeStrategy`: a pre-timed plan from Webster's method using design-hour
  volumes, the way most urban signals are still timed.
* :class:`ActuatedStrategy`: NEMA-style fully actuated control. Green is extended while
  vehicles keep actuating the detectors and gaps out after the passage time, with max-out
  enforced by the controller.
* :class:`MaxPressureStrategy`: Varaiya's (2013) max-pressure policy on estimated queues,
  with a hysteresis margin that accounts for the lost time of each switch.
"""

from __future__ import annotations

from typing import ClassVar

from jevflow.control.base import ControlStrategy, Decision, DecisionContext
from jevflow.domain.enums import ControlAction, DecisionSource, Phase
from jevflow.domain.observation import TrafficObservation
from jevflow.domain.timing import SignalTiming, webster_cycle
from jevflow.simulation.scenario import Scenario

SATURATION_FLOW_VPH = 1800.0
PERMISSIVE_LEFT_SATURATION_VPH = 700.0
START_UP_LOST_S = 2.0


def conflicting_demand(obs: TrafficObservation) -> bool:
    other = obs.signal.current_phase.other
    if obs.pedestrian_calls[other]:
        return True
    return any(
        obs.approaches[a].estimated_queue_vehicles >= 0.5 or obs.approaches[a].stop_line_occupancy_percent > 0
        for a in other.approaches
    )


def _keep(reason: str) -> Decision:
    return Decision(ControlAction.KEEP_CURRENT_PHASE, DecisionSource.STRATEGY, reason)


def _switch(reason: str) -> Decision:
    return Decision(ControlAction.SWITCH_PHASE, DecisionSource.STRATEGY, reason)


class FixedTimeStrategy(ControlStrategy):
    key: ClassVar[str] = "fixed_time"
    name: ClassVar[str] = "Fixed-Time (Webster)"
    description: ClassVar[str] = "Pre-timed plan from design-hour volumes. Ignores detectors entirely."
    decision_interval_s = 0.5

    def __init__(self, scenario: Scenario, timing: SignalTiming) -> None:
        self.cycle_s, self.greens = self.plan(scenario, timing)

    @staticmethod
    def plan(scenario: Scenario, timing: SignalTiming) -> tuple[float, dict[Phase, float]]:
        ratios: dict[Phase, float] = {}
        for phase in Phase:
            critical = 0.0
            for approach in phase.approaches:
                demand = scenario.demand[approach]
                peak = demand.peak_vph
                through = peak * (demand.turning.through + demand.turning.right) / SATURATION_FLOW_VPH
                left = peak * demand.turning.left / PERMISSIVE_LEFT_SATURATION_VPH
                critical = max(critical, through, left)
            ratios[phase] = critical
        lost = timing.clearance_s + START_UP_LOST_S - 1.0
        cycle = min(max(webster_cycle(list(ratios.values()), lost), 50.0), 150.0)
        total = sum(ratios.values()) or 1.0
        available = cycle - len(Phase) * timing.clearance_s
        needs_peds = any(rate > 0 for rate in scenario.pedestrians_per_hour.values())
        floor = max(timing.min_green_s, timing.ped_min_green_s if needs_peds else 0.0)
        greens = {
            phase: min(max(available * ratios[phase] / total, floor), timing.max_green_s) for phase in Phase
        }
        return cycle, greens

    async def decide(self, context: DecisionContext) -> Decision:
        signal = context.observation.signal
        planned = self.greens[signal.current_phase]
        if signal.phase_elapsed_seconds >= planned:
            return _switch(f"planned green of {planned:.0f}s elapsed (cycle {self.cycle_s:.0f}s)")
        return _keep(f"{planned - signal.phase_elapsed_seconds:.0f}s of planned green remain")


class ActuatedStrategy(ControlStrategy):
    key: ClassVar[str] = "actuated"
    name: ClassVar[str] = "Actuated (gap-out)"
    description: ClassVar[str] = "Extends green while detectors keep firing; gaps out after the passage time."
    decision_interval_s = 0.5

    def __init__(self, passage_time_s: float = 3.0) -> None:
        self.passage_time_s = passage_time_s

    async def decide(self, context: DecisionContext) -> Decision:
        obs = context.observation
        if not conflicting_demand(obs):
            return _keep("no conflicting call: rest in green")
        served = obs.signal.current_phase.approaches
        gap = min(obs.approaches[a].seconds_since_last_actuation for a in served)
        if gap >= self.passage_time_s:
            return _switch(f"gap-out: {gap:.1f}s without an actuation on the green approaches")
        return _keep(f"extending: last actuation {gap:.1f}s ago")


class MaxPressureStrategy(ControlStrategy):
    key: ClassVar[str] = "max_pressure"
    name: ClassVar[str] = "Max-Pressure"
    description: ClassVar[str] = "Serves the phase with the largest estimated queue (Varaiya, 2013)."
    decision_interval_s = 2.0

    def __init__(self, switch_margin_veh: float = 3.0, ped_weight_veh: float = 3.0) -> None:
        self.switch_margin_veh = switch_margin_veh
        self.ped_weight_veh = ped_weight_veh

    def pressure(self, obs: TrafficObservation, phase: Phase) -> float:
        queue = obs.phase_queue(phase)
        return queue + (self.ped_weight_veh if obs.pedestrian_calls[phase] else 0.0)

    async def decide(self, context: DecisionContext) -> Decision:
        obs = context.observation
        current = obs.signal.current_phase
        mine, theirs = self.pressure(obs, current), self.pressure(obs, current.other)
        if theirs > mine + self.switch_margin_veh:
            return _switch(f"pressure {theirs:.1f} vs {mine:.1f} veh favours {current.other.label}")
        return _keep(f"pressure {mine:.1f} vs {theirs:.1f} veh: stay on {current.label}")
