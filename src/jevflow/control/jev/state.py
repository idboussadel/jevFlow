"""Builds the Jev ``state``: a validated, expert-readable snapshot of the intersection.

Following TypeSafe's guidance ("code calculates, Jev judges"), deterministic code precomputes
the comparisons an engineer would glance at: queue served vs waiting, longest red wait, green
time left before max-out, and switching cost. Jev spends its judgement on the trade-off, not on
arithmetic. Every value comes from the detector layer; nothing here is ground truth.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jevflow.control.base import DecisionContext
from jevflow.domain.enums import Approach, ControlAction, Phase
from jevflow.domain.observation import ApproachMeasurement, TrafficObservation

INTERSECTION_ID = "jevflow_main_and_1st"
START_UP_LOST_S = 2.0


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class JevSignal(_Model):
    current_phase: str
    phase_elapsed_seconds: float
    minimum_green_seconds: float
    maximum_green_seconds: float
    green_remaining_before_max_out_seconds: float
    extension_seconds: float
    yellow_seconds: float
    all_red_seconds: float
    switch_lost_time_seconds: float = Field(description="Yellow + all-red + start-up time lost by switching")
    time_since_north_last_received_green: float
    time_since_south_last_received_green: float
    time_since_east_last_received_green: float
    time_since_west_last_received_green: float


class JevApproach(_Model):
    currently_green: bool
    volume_vehicles_30s: int
    departures_vehicles_30s: int
    occupancy_percent: float
    average_speed_kmh: float
    estimated_queue_vehicles: float
    estimated_left_turn_queue_vehicles: float
    average_delay_seconds: float
    seconds_since_last_detection: float
    queue_spillback_detected: bool
    detector_health: str


class JevPhaseSummary(_Model):
    status: str
    total_queue_vehicles: float
    arrivals_30s: int
    departures_30s: int
    worst_average_delay_seconds: float
    longest_wait_since_green_seconds: float
    pedestrians_waiting: bool
    spillback: bool


class JevSpecialConditions(_Model):
    emergency_vehicle_present: bool
    emergency_vehicle_approach: str | None
    emergency_vehicle_distance_m: float | None
    pedestrian_crossing_request: bool
    queue_spillback_detected: bool
    detectors_degraded: list[str]


class JevObjective(_Model):
    reduce_delay: bool = True
    reduce_queue_length: bool = True
    prevent_long_waits: bool = True
    avoid_queue_spillback: bool = True
    avoid_unnecessary_switching: bool = True


class JevTrafficState(_Model):
    intersection_id: str
    time_seconds: float
    signal: JevSignal
    approaches: dict[str, JevApproach]
    phase_summary: dict[str, JevPhaseSummary]
    special_conditions: JevSpecialConditions
    legal_actions: list[str]
    objective: JevObjective = JevObjective()


def action_label(action: ControlAction, current: Phase) -> str:
    """Human-meaningful option names, matching the phase they lead to."""
    if action is ControlAction.SWITCH_PHASE:
        return f"switch_to_{current.other.value}"
    return action.value


def _approach(m: ApproachMeasurement, green: bool) -> JevApproach:
    return JevApproach(
        currently_green=green,
        volume_vehicles_30s=m.volume_vehicles_30s,
        departures_vehicles_30s=m.departures_30s,
        occupancy_percent=m.occupancy_percent,
        average_speed_kmh=m.average_speed_kmh,
        estimated_queue_vehicles=m.estimated_queue_vehicles,
        estimated_left_turn_queue_vehicles=m.estimated_left_turn_queue_vehicles,
        average_delay_seconds=m.average_delay_seconds,
        seconds_since_last_detection=m.seconds_since_last_actuation,
        queue_spillback_detected=m.queue_spillback_detected,
        detector_health=m.detector_health.value,
    )


def _phase_summary(obs: TrafficObservation, phase: Phase) -> JevPhaseSummary:
    ms = [obs.approaches[a] for a in phase.approaches]
    return JevPhaseSummary(
        status="green_now" if obs.signal.current_phase is phase else "waiting_on_red",
        total_queue_vehicles=round(sum(m.estimated_queue_vehicles for m in ms), 1),
        arrivals_30s=sum(m.volume_vehicles_30s for m in ms),
        departures_30s=sum(m.departures_30s for m in ms),
        worst_average_delay_seconds=max(m.average_delay_seconds for m in ms),
        longest_wait_since_green_seconds=obs.longest_wait(phase),
        pedestrians_waiting=obs.pedestrian_calls[phase],
        spillback=any(m.queue_spillback_detected for m in ms),
    )


def build_state(context: DecisionContext) -> JevTrafficState:
    obs = context.observation
    sig = obs.signal
    timing = context.timing
    current = sig.current_phase
    since = sig.time_since_last_green
    degraded = sorted({d for m in obs.approaches.values() for d in m.faulty_detectors})
    return JevTrafficState(
        intersection_id=INTERSECTION_ID,
        time_seconds=obs.time_seconds,
        signal=JevSignal(
            current_phase=f"{current.value}_green",
            phase_elapsed_seconds=sig.phase_elapsed_seconds,
            minimum_green_seconds=sig.minimum_green_seconds,
            maximum_green_seconds=sig.maximum_green_seconds,
            green_remaining_before_max_out_seconds=round(
                max(0.0, sig.maximum_green_seconds - sig.phase_elapsed_seconds), 1
            ),
            extension_seconds=timing.extension_s,
            yellow_seconds=sig.yellow_seconds,
            all_red_seconds=sig.all_red_seconds,
            switch_lost_time_seconds=sig.yellow_seconds + sig.all_red_seconds + START_UP_LOST_S,
            time_since_north_last_received_green=since[Approach.NORTH],
            time_since_south_last_received_green=since[Approach.SOUTH],
            time_since_east_last_received_green=since[Approach.EAST],
            time_since_west_last_received_green=since[Approach.WEST],
        ),
        approaches={a.value: _approach(obs.approaches[a], a.phase is current) for a in Approach.ordered()},
        phase_summary={p.value: _phase_summary(obs, p) for p in (current, current.other)},
        special_conditions=JevSpecialConditions(
            emergency_vehicle_present=obs.emergency_vehicle_approach is not None,
            emergency_vehicle_approach=obs.emergency_vehicle_approach.value
            if obs.emergency_vehicle_approach
            else None,
            emergency_vehicle_distance_m=obs.emergency_vehicle_distance_m,
            pedestrian_crossing_request=any(obs.pedestrian_calls.values()),
            queue_spillback_detected=any(m.queue_spillback_detected for m in obs.approaches.values()),
            detectors_degraded=degraded,
        ),
        legal_actions=[action_label(a, current) for a in context.legal_actions],
    )
