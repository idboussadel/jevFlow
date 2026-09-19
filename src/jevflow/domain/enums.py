"""Core vocabulary of the intersection: approaches, movements, phases and control actions.

Traffic drives on the right. Coordinates are metres with the intersection centre at the origin,
``+x`` pointing east and ``+y`` pointing north.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Approach(StrEnum):
    """An intersection leg, named after where its inbound traffic comes *from*."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def ordinal(self) -> int:
        """Stable position (N, S, E, W) used to index NumPy arrays."""
        return _APPROACH_ORDER.index(self)

    @property
    def opposite(self) -> Approach:
        return _OPPOSITE[self]

    @property
    def phase(self) -> Phase:
        return Phase.NORTH_SOUTH if self in (Approach.NORTH, Approach.SOUTH) else Phase.EAST_WEST

    @property
    def travel_direction(self) -> tuple[float, float]:
        """Unit vector of inbound travel (traffic from the north travels south)."""
        return _TRAVEL[self]

    @classmethod
    def ordered(cls) -> tuple[Approach, ...]:
        return _APPROACH_ORDER

    @classmethod
    def from_index(cls, index: int) -> Approach:
        return _APPROACH_ORDER[index]

    @classmethod
    def from_outward_direction(cls, dx: float, dy: float) -> Approach:
        """The leg a vehicle leaves through when travelling along ``(dx, dy)``."""
        for approach, (tx, ty) in _TRAVEL.items():
            if round(-tx) == round(dx) and round(-ty) == round(dy):
                return approach
        raise ValueError(f"({dx}, {dy}) is not an axis-aligned unit vector")


_APPROACH_ORDER = (Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST)
_OPPOSITE = {
    Approach.NORTH: Approach.SOUTH,
    Approach.SOUTH: Approach.NORTH,
    Approach.EAST: Approach.WEST,
    Approach.WEST: Approach.EAST,
}
_TRAVEL = {
    Approach.NORTH: (0.0, -1.0),
    Approach.SOUTH: (0.0, 1.0),
    Approach.EAST: (-1.0, 0.0),
    Approach.WEST: (1.0, 0.0),
}


class Movement(IntEnum):
    """Turning movement. Integer-valued so it can live in NumPy arrays."""

    LEFT = 0
    THROUGH = 1
    RIGHT = 2


class Phase(StrEnum):
    """Two-phase operation: each phase serves both opposing approaches, lefts permissive."""

    NORTH_SOUTH = "north_south"
    EAST_WEST = "east_west"

    @property
    def other(self) -> Phase:
        return Phase.EAST_WEST if self is Phase.NORTH_SOUTH else Phase.NORTH_SOUTH

    @property
    def approaches(self) -> tuple[Approach, Approach]:
        if self is Phase.NORTH_SOUTH:
            return (Approach.NORTH, Approach.SOUTH)
        return (Approach.EAST, Approach.WEST)

    @property
    def label(self) -> str:
        return "north-south" if self is Phase.NORTH_SOUTH else "east-west"


class Interval(StrEnum):
    """Signal interval within a phase transition sequence."""

    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


class SignalColor(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class VehicleClass(IntEnum):
    CAR = 0
    VAN = 1
    BUS = 2
    TRUCK = 3
    EMERGENCY = 4


class ControlAction(StrEnum):
    """Commands a control strategy may issue to the signal controller."""

    KEEP_CURRENT_PHASE = "keep_current_phase"
    EXTEND_CURRENT_GREEN = "extend_current_green"
    SWITCH_PHASE = "switch_phase"


class DecisionSource(StrEnum):
    """Who produced a decision, for auditability."""

    STRATEGY = "strategy"  # a deterministic baseline controller
    JEV = "jev"
    FALLBACK = "fallback"  # Jev unavailable (API error / open circuit)
    SAFETY = "safety"  # forced by the signal controller (max-out, preemption)
