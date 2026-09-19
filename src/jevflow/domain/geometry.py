"""Physical layout of a four-leg, two-lanes-per-direction intersection.

Each inbound approach has an exclusive left-turn lane (lane 0, nearest the centre line) and a
shared through/right lane (lane 1). Each outbound leg has two lanes. Turning paths through the
junction box are cubic Bézier curves sampled into arc-length lookup tables so vehicle poses can
be computed with vectorised interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from jevflow.domain.enums import Approach, Movement

FloatArray = npt.NDArray[np.float64]

INBOUND_LANES = 2
LEFT_LANE = 0
THROUGH_RIGHT_LANE = 1
_PATH_SAMPLES = 96


def lane_for_movement(movement: Movement) -> int:
    return LEFT_LANE if movement is Movement.LEFT else THROUGH_RIGHT_LANE


def outbound_lane_for_movement(movement: Movement) -> int:
    """Lefts enter the inner lane, throughs and rights the kerb lane."""
    return 0 if movement is Movement.LEFT else 1


def exit_approach(approach: Approach, movement: Movement) -> Approach:
    dx, dy = approach.travel_direction
    match movement:
        case Movement.THROUGH:
            return Approach.from_outward_direction(dx, dy)
        case Movement.RIGHT:
            return Approach.from_outward_direction(dy, -dx)
        case Movement.LEFT:
            return Approach.from_outward_direction(-dy, dx)


@dataclass(frozen=True, slots=True)
class TurnPath:
    """Arc-length parameterised path across the junction box."""

    approach: Approach
    movement: Movement
    s: FloatArray  # cumulative arc length at each sample
    x: FloatArray
    y: FloatArray
    heading: FloatArray  # radians, 0 = east, counter-clockwise

    @property
    def length(self) -> float:
        return float(self.s[-1])


@dataclass(frozen=True)
class IntersectionGeometry:
    lane_width_m: float = 3.5
    approach_length_m: float = 250.0
    exit_length_m: float = 120.0
    stop_line_offset_m: float = 12.0  # distance from centre to stop line
    crosswalk_width_m: float = 4.0
    speed_limit_kmh: float = 50.0
    _paths: dict[tuple[Approach, Movement], TurnPath] = field(default_factory=dict, init=False, repr=False)

    @property
    def speed_limit_ms(self) -> float:
        return self.speed_limit_kmh / 3.6

    @property
    def road_half_width_m(self) -> float:
        return self.lane_width_m * INBOUND_LANES

    @property
    def crossing_distance_m(self) -> float:
        """Kerb-to-kerb distance a pedestrian walks across one leg."""
        return 2 * self.road_half_width_m

    @property
    def crosswalk_centre_offset_m(self) -> float:
        return self.road_half_width_m + 0.5 + self.crosswalk_width_m / 2

    # ------------------------------------------------------------------ positions
    def lane_offset(self, lane: int) -> float:
        """Lateral distance of a lane centre from the road centre line."""
        return (lane + 0.5) * self.lane_width_m

    def inbound_pose(
        self, approach: Approach, lane: int, pos: FloatArray
    ) -> tuple[FloatArray, FloatArray, float]:
        """World coordinates for inbound positions (0 at the approach entry, ``L`` at the stop line)."""
        dx, dy = approach.travel_direction
        rx, ry = dy, -dx  # right-hand normal of the travel direction
        back = self.stop_line_offset_m + self.approach_length_m - pos
        off = self.lane_offset(lane)
        return -dx * back + rx * off, -dy * back + ry * off, float(np.arctan2(dy, dx))

    def outbound_pose(
        self, leg: Approach, lane: int, pos: FloatArray
    ) -> tuple[FloatArray, FloatArray, float]:
        """World coordinates on an outbound leg (0 just beyond the junction box)."""
        ox, oy = -leg.travel_direction[0], -leg.travel_direction[1]  # outward direction
        rx, ry = oy, -ox
        dist = self.stop_line_offset_m + pos
        off = self.lane_offset(lane)
        return ox * dist + rx * off, oy * dist + ry * off, float(np.arctan2(oy, ox))

    # ------------------------------------------------------------------ turn paths
    def path(self, approach: Approach, movement: Movement) -> TurnPath:
        key = (approach, movement)
        if key not in self._paths:
            self._paths[key] = self._build_path(approach, movement)
        return self._paths[key]

    @cached_property
    def path_lengths(self) -> FloatArray:
        """Box path length indexed by ``approach.ordinal * 3 + movement``."""
        return np.array(
            [self.path(a, m).length for a in Approach.ordered() for m in Movement], dtype=np.float64
        )

    def _build_path(self, approach: Approach, movement: Movement) -> TurnPath:
        in_lane = lane_for_movement(movement)
        dx, dy = approach.travel_direction
        x0, y0, _ = self.inbound_pose(approach, in_lane, np.array([self.approach_length_m]))
        leg = exit_approach(approach, movement)
        x3, y3, _ = self.outbound_pose(leg, outbound_lane_for_movement(movement), np.array([0.0]))
        p0 = np.array([x0[0], y0[0]])
        p3 = np.array([x3[0], y3[0]])
        ex, ey = -leg.travel_direction[0], -leg.travel_direction[1]
        chord = float(np.linalg.norm(p3 - p0))
        # Straight paths use handles at thirds (a uniform line); turns approximate a circular arc.
        k = chord / 3 if movement is Movement.THROUGH else 0.5523 * chord / np.sqrt(2)
        p1 = p0 + k * np.array([dx, dy])
        p2 = p3 - k * np.array([ex, ey])

        t = np.linspace(0.0, 1.0, _PATH_SAMPLES)[:, None]
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        d1 = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        heading = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
        return TurnPath(approach, movement, s, pts[:, 0].copy(), pts[:, 1].copy(), heading)
