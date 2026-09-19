"""Structure-of-arrays storage for vehicles.

Keeping each attribute in its own contiguous NumPy array lets the engine update every
vehicle with a handful of vectorised operations per tick, instead of a Python loop over
objects. Slots are recycled; ``active`` marks which ones hold a vehicle.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from jevflow.simulation.demand import Arrival


class Stage(IntEnum):
    INBOUND = 0  # on an approach lane, before the stop line
    BOX = 1  # inside the junction
    OUTBOUND = 2  # on an exit leg


class YellowDecision(IntEnum):
    """Choice made at yellow onset. Values ``>= GO`` proceed through the stop line."""

    NONE = 0
    STOP = 1
    GO = 2
    SNEAK = 3  # permissive left already waiting at the line: legally clears on yellow/all-red


_FLOAT_FIELDS = (
    "s", "v", "acc", "length", "v0", "T", "a", "b", "s0", "reaction", "crit_gap",
    "yellow_ttsl", "max_stop_decel", "launch_timer", "arrival_time", "crossed_time",
    "ff_time", "path_len", "turn_speed", "wait_total",
)  # fmt: skip
_INT_FIELDS: dict[str, Any] = {
    "vid": np.int64, "cls": np.int8, "approach": np.int8, "lane": np.int8, "movement": np.int8,
    "stage": np.int8, "out_link": np.int16, "yellow": np.int8, "n_stops": np.int16, "miss_mask": np.int8,
}  # fmt: skip
_BOOL_FIELDS = ("active", "was_moving")


class Fleet:
    s: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    acc: npt.NDArray[np.float64]
    length: npt.NDArray[np.float64]
    v0: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]
    s0: npt.NDArray[np.float64]
    reaction: npt.NDArray[np.float64]
    crit_gap: npt.NDArray[np.float64]
    yellow_ttsl: npt.NDArray[np.float64]
    max_stop_decel: npt.NDArray[np.float64]
    launch_timer: npt.NDArray[np.float64]
    arrival_time: npt.NDArray[np.float64]
    crossed_time: npt.NDArray[np.float64]
    ff_time: npt.NDArray[np.float64]
    path_len: npt.NDArray[np.float64]
    turn_speed: npt.NDArray[np.float64]
    wait_total: npt.NDArray[np.float64]
    vid: npt.NDArray[np.int64]
    cls: npt.NDArray[np.int8]
    approach: npt.NDArray[np.int8]
    lane: npt.NDArray[np.int8]
    movement: npt.NDArray[np.int8]
    stage: npt.NDArray[np.int8]
    out_link: npt.NDArray[np.int16]
    yellow: npt.NDArray[np.int8]
    n_stops: npt.NDArray[np.int16]
    miss_mask: npt.NDArray[np.int8]
    active: npt.NDArray[np.bool_]
    was_moving: npt.NDArray[np.bool_]

    def __init__(self, capacity: int = 512) -> None:
        self._capacity = 0
        self._allocate(capacity)
        self._next_vid = 1

    def _allocate(self, capacity: int) -> None:
        old = self._capacity
        for name in _FLOAT_FIELDS:
            self._resize(name, capacity, np.float64, old)
        for name, dtype in _INT_FIELDS.items():
            self._resize(name, capacity, dtype, old)
        for name in _BOOL_FIELDS:
            self._resize(name, capacity, np.bool_, old)
        self._capacity = capacity

    def _resize(self, name: str, capacity: int, dtype: Any, old: int) -> None:
        fresh = np.zeros(capacity, dtype=dtype)
        if old:
            fresh[:old] = getattr(self, name)
        setattr(self, name, fresh)

    @property
    def indices(self) -> npt.NDArray[np.intp]:
        return np.flatnonzero(self.active)

    def __len__(self) -> int:
        return int(self.active.sum())

    def add(
        self,
        arrival: Arrival,
        *,
        speed: float,
        out_link: int,
        path_len: float,
        turn_speed: float,
        ff_time: float,
    ) -> int:
        free = np.flatnonzero(~self.active)
        if free.size == 0:
            self._allocate(self._capacity * 2)
            free = np.flatnonzero(~self.active)
        i = int(free[0])
        d = arrival.driver
        self.vid[i] = self._next_vid
        self._next_vid += 1
        self.active[i] = True
        self.s[i] = 0.0
        self.v[i] = speed
        self.acc[i] = 0.0
        self.length[i] = d.length
        self.v0[i] = d.desired_speed
        self.T[i] = d.time_gap
        self.a[i] = d.max_accel
        self.b[i] = d.comfort_decel
        self.s0[i] = d.min_gap
        self.reaction[i] = d.reaction_time
        self.crit_gap[i] = d.critical_gap
        self.yellow_ttsl[i] = d.yellow_ttsl
        self.max_stop_decel[i] = d.max_stop_decel
        self.launch_timer[i] = -1.0
        self.arrival_time[i] = arrival.time
        self.crossed_time[i] = np.nan
        self.ff_time[i] = ff_time
        self.path_len[i] = path_len
        self.turn_speed[i] = turn_speed
        self.wait_total[i] = 0.0
        self.cls[i] = int(d.vclass)
        self.approach[i] = arrival.approach.ordinal
        self.lane[i] = arrival.lane
        self.movement[i] = int(arrival.movement)
        self.stage[i] = Stage.INBOUND
        self.out_link[i] = out_link
        self.yellow[i] = YellowDecision.NONE
        self.n_stops[i] = 0
        self.miss_mask[i] = arrival.missed_by
        self.was_moving[i] = speed > 3.0
        return i

    def remove(self, idx: npt.NDArray[np.intp]) -> None:
        self.active[idx] = False
