"""Pedestrians: push-button calls, waiting at the kerb, crossing on WALK.

Each leg has one crosswalk. It is served by the phase that runs *parallel* to it, so the
crosswalk across the north leg walks with east-west traffic. Turning vehicles yield to anyone
on the crosswalk they turn across.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from jevflow.domain.enums import Approach
from jevflow.domain.geometry import IntersectionGeometry
from jevflow.simulation.demand import PedestrianArrival
from jevflow.simulation.signal import PedSignal, SignalController

KERB_MARGIN_M = 1.5


class PedState(StrEnum):
    WAITING = "waiting"
    CROSSING = "crossing"


@dataclass(slots=True)
class Pedestrian:
    pid: int
    leg: Approach
    side: int
    speed: float
    arrived_at: float
    state: PedState = PedState.WAITING
    progress: float = 0.0
    started_at: float = 0.0


@dataclass(slots=True)
class PedestrianStats:
    served: int = 0
    total_wait_s: float = 0.0
    max_wait_s: float = 0.0
    waits: list[float] = field(default_factory=list)


class PedestrianModel:
    def __init__(self, arrivals: list[PedestrianArrival], geometry: IntersectionGeometry) -> None:
        self._pending = deque(arrivals)
        self._geometry = geometry
        self._crossing_length = geometry.crossing_distance_m + 2 * KERB_MARGIN_M
        self.active: list[Pedestrian] = []
        self.stats = PedestrianStats()
        self._next_id = 1

    def step(self, t: float, dt: float, signal: SignalController) -> None:
        while self._pending and self._pending[0].time <= t:
            arrival = self._pending.popleft()
            self.active.append(Pedestrian(self._next_id, arrival.leg, arrival.side, arrival.speed, t))
            self._next_id += 1
            signal.call_pedestrian(arrival.leg)

        finished: list[Pedestrian] = []
        for ped in self.active:
            if ped.state is PedState.WAITING:
                if signal.ped_signal(ped.leg) is PedSignal.WALK:
                    ped.state = PedState.CROSSING
                    ped.started_at = t
                    self._record_wait(t - ped.arrived_at)
                else:
                    signal.call_pedestrian(ped.leg)  # keep the call registered
            else:
                ped.progress += ped.speed * dt
                if ped.progress >= self._crossing_length:
                    finished.append(ped)
        for ped in finished:
            self.active.remove(ped)

    def _record_wait(self, wait: float) -> None:
        self.stats.served += 1
        self.stats.total_wait_s += wait
        self.stats.max_wait_s = max(self.stats.max_wait_s, wait)
        self.stats.waits.append(wait)

    def crosswalk_busy(self, leg: Approach) -> bool:
        """True while someone is on the roadway portion of the crosswalk across ``leg``."""
        return any(
            p.leg is leg
            and p.state is PedState.CROSSING
            and KERB_MARGIN_M * 0.5 < p.progress < self._crossing_length - KERB_MARGIN_M * 0.5
            for p in self.active
        )

    def waiting_count(self, leg: Approach) -> int:
        return sum(1 for p in self.active if p.leg is leg and p.state is PedState.WAITING)

    def poses(self) -> list[tuple[int, float, float, int]]:
        """(id, x, y, crossing) for rendering."""
        out: list[tuple[int, float, float, int]] = []
        c = self._geometry.crosswalk_centre_offset_m
        half = self._crossing_length / 2
        for p in self.active:
            ox, oy = -p.leg.travel_direction[0], -p.leg.travel_direction[1]  # outward along the leg
            lx, ly = oy, -ox  # across the leg
            sign = 1.0 if p.side == 0 else -1.0
            jitter = ((p.pid * 7919) % 100) / 100.0 - 0.5
            if p.state is PedState.WAITING:
                lateral = sign * -(half + 0.6 + 0.4 * abs(jitter))
                along = c + jitter * 2.2
                crossing = 0
            else:
                lateral = sign * (-half + p.progress)
                along = c + jitter * 1.6
                crossing = 1
            out.append((p.pid, ox * along + lx * lateral, oy * along + ly * lateral, crossing))
        return out
