"""Signal timing parameters and the traffic-engineering formulas that derive them.

References
----------
* ITE (2020) *Guidelines for Determining Traffic Signal Change and Clearance Intervals*:
  ``Y = t + v / (2a + 2Gg)`` and ``R = (W + L) / v``.
* MUTCD 4E.06: pedestrian clearance at 3.5 ft/s (1.07 m/s), minimum walk 7 s.
* Webster (1958) optimum cycle ``C0 = (1.5L + 5) / (1 - Y)``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, computed_field

GRAVITY = 9.81


def ite_yellow_interval(
    approach_speed_ms: float,
    perception_reaction_s: float = 1.0,
    deceleration_ms2: float = 3.05,
    grade: float = 0.0,
) -> float:
    """Yellow change interval (s) from the kinematic ITE formula."""
    return perception_reaction_s + approach_speed_ms / (2 * deceleration_ms2 + 2 * grade * GRAVITY)


def ite_all_red_interval(
    approach_speed_ms: float, crossing_width_m: float, vehicle_length_m: float = 6.0
) -> float:
    """Red clearance interval (s): time for a vehicle entering at the end of yellow to clear."""
    return (crossing_width_m + vehicle_length_m) / approach_speed_ms


def pedestrian_clearance(crossing_distance_m: float, walking_speed_ms: float = 1.07) -> float:
    return crossing_distance_m / walking_speed_ms


def round_up(value: float, step: float = 0.5) -> float:
    return math.ceil(value / step) * step


class SignalTiming(BaseModel):
    """Controller timing plan shared by every control strategy."""

    model_config = ConfigDict(frozen=True)

    min_green_s: float = Field(10.0, ge=4.0, description="Minimum green before a phase may terminate")
    max_green_s: float = Field(60.0, gt=0.0, description="Max-out: green is terminated after this")
    yellow_s: float = Field(3.5, ge=3.0, le=6.0)
    all_red_s: float = Field(2.0, ge=0.0, le=6.0)
    extension_s: float = Field(5.0, gt=0.0, description="Green extension granted by extend_current_green")
    walk_s: float = Field(7.0, ge=4.0)
    ped_clearance_s: float = Field(13.0, ge=0.0)
    preempt_min_green_s: float = Field(
        5.0, ge=0.0, description="Shortest green before an EV preemption switch"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clearance_s(self) -> float:
        """Yellow plus all-red: the time lost every time the phase changes."""
        return self.yellow_s + self.all_red_s

    @property
    def ped_min_green_s(self) -> float:
        """Green needed to serve a pedestrian call (walk + flashing don't walk)."""
        return self.walk_s + self.ped_clearance_s

    @classmethod
    def engineered(
        cls,
        speed_limit_ms: float,
        crossing_width_m: float,
        crossing_distance_m: float,
        **overrides: float,
    ) -> SignalTiming:
        """Timing derived from geometry and speed, rounded up to 0.5 s like field practice."""
        return cls(
            yellow_s=round_up(ite_yellow_interval(speed_limit_ms)),
            all_red_s=round_up(ite_all_red_interval(speed_limit_ms, crossing_width_m)),
            ped_clearance_s=round_up(pedestrian_clearance(crossing_distance_m)),
            **overrides,
        )


def webster_cycle(critical_flow_ratios: list[float], lost_time_per_phase_s: float) -> float:
    """Webster's delay-minimising cycle length for the given critical ``v/s`` ratios."""
    total_lost = lost_time_per_phase_s * len(critical_flow_ratios)
    y_total = min(sum(critical_flow_ratios), 0.92)  # beyond ~0.9 the formula diverges
    return (1.5 * total_lost + 5.0) / (1.0 - y_total)
