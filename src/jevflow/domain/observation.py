"""What a traffic controller can know: detector-derived measurements plus controller state.

Nothing in this module is ground truth. Every number is produced by the virtual detector layer
and the traffic-state aggregator, so it carries the same blind spots a field controller has:
missed actuations, queues longer than the detector setback, faulty loops, quantised speeds.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jevflow.domain.enums import Approach, Interval, Phase, SignalColor


class DetectorHealth(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"  # at least one loop flagged; estimates use fallbacks


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ApproachMeasurement(_Frozen):
    """Aggregated detector data for one approach over the rolling window."""

    volume_vehicles_30s: int = Field(ge=0, description="Arrivals counted at the advance loops")
    departures_30s: int = Field(ge=0, description="Vehicles counted crossing the stop-line loops")
    occupancy_percent: float = Field(ge=0, le=100, description="Advance-loop time occupancy")
    stop_line_occupancy_percent: float = Field(ge=0, le=100)
    average_speed_kmh: float = Field(ge=0, description="Speed-trap harmonic mean, or single-loop estimate")
    estimated_queue_vehicles: float = Field(ge=0)
    estimated_left_turn_queue_vehicles: float = Field(ge=0)
    average_delay_seconds: float = Field(ge=0)
    queue_spillback_detected: bool = False
    detector_health: DetectorHealth = DetectorHealth.OK
    faulty_detectors: tuple[str, ...] = ()
    seconds_since_last_actuation: float = Field(ge=0, description="Gap since any loop on the approach fired")


class SignalStatus(_Frozen):
    """Signal controller state; exact, because the controller owns it."""

    current_phase: Phase
    interval: Interval
    colors: dict[Approach, SignalColor]
    interval_elapsed_seconds: float = Field(ge=0)
    phase_elapsed_seconds: float = Field(ge=0, description="Green time elapsed in the current phase")
    minimum_green_seconds: float
    maximum_green_seconds: float
    yellow_seconds: float
    all_red_seconds: float
    extension_remaining_seconds: float = Field(ge=0)
    time_since_last_green: dict[Approach, float]
    preempted_by: Approach | None = None


class TrafficObservation(_Frozen):
    """A complete controller-side snapshot at one instant."""

    time_seconds: float = Field(ge=0)
    signal: SignalStatus
    approaches: dict[Approach, ApproachMeasurement]
    pedestrian_calls: dict[Phase, bool]
    emergency_vehicle_approach: Approach | None = None
    emergency_vehicle_distance_m: float | None = None

    def phase_queue(self, phase: Phase) -> float:
        return sum(self.approaches[a].estimated_queue_vehicles for a in phase.approaches)

    def longest_wait(self, phase: Phase) -> float:
        return max(self.signal.time_since_last_green[a] for a in phase.approaches)
