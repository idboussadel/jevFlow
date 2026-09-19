"""Virtual inductive-loop detectors.

The engine knows every vehicle's exact position; a loop only knows whether *something* is over
it. Each detector samples presence at the controller's 10 Hz scan rate, so occupancy and speed
inherit the same quantisation a field controller has. On top of that:

* **Missed vehicles**: each vehicle carries a pre-drawn mask of loops that will not see it
  (low-magnetic-signature vehicles, cross-talk).
* **False actuations**: random short pulses (splash-over, induced noise).
* **Faults**: loops can be stuck ON, dead, or chattering for scheduled windows.

Per inbound lane there are three loops, in the NEMA-style layout:

* ``stop``: 2 m loop just ahead of the stop line (presence, departures, occupancy);
* ``advance``: 2 m loop plus a second loop 4 m downstream forming a speed trap;
* ``entry``: 2 m loop near the upstream end, used to detect queue spillback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from jevflow.domain.enums import Approach
from jevflow.domain.geometry import INBOUND_LANES
from jevflow.simulation.engine import SimulationEngine
from jevflow.simulation.fleet import Stage
from jevflow.simulation.scenario import DetectorKind, FaultEvent, SensorProfile

LOOP_LENGTH_M = 2.0
SPEED_TRAP_SPACING_M = 4.0
ADVANCE_SETBACK_M = 70.0  # typical dilemma-zone setback at 50 km/h
ENTRY_SETBACK_FROM_START_M = 12.0


class LoopRole(IntEnum):
    """Bit position used in each vehicle's miss mask."""

    STOP = 0
    ADVANCE = 1
    ENTRY = 2


@dataclass(slots=True)
class LoopDetector:
    """One physical loop: its footprint on the lane and its measurement history."""

    detector_id: str
    approach: Approach
    lane: int
    role: LoopRole
    start_m: float  # upstream edge, lane coordinate
    end_m: float  # downstream edge
    present: bool = False
    fault: str | None = None
    rising_edges: int = 0
    last_on_time: float = -1e9  # last rising edge
    last_present_time: float = -1e9  # last scan with the loop occupied
    on_since: float | None = None
    _false_until: float = 0.0
    _chatter_phase: float = 0.0
    history: deque[tuple[float, bool, bool]] = field(default_factory=lambda: deque(maxlen=1200))
    """(time, presence, rising-edge) samples; 120 s at 10 Hz."""

    @property
    def centre_m(self) -> float:
        return 0.5 * (self.start_m + self.end_m)


@dataclass(frozen=True, slots=True)
class SpeedSample:
    time: float
    speed_ms: float


class DetectorArray:
    """All loops at the intersection, sampled every tick from the engine's ground truth."""

    def __init__(self, engine: SimulationEngine, profile: SensorProfile) -> None:
        self._engine = engine
        self._profile = profile
        self._rng = engine.streams.get("detector-noise")
        L = engine.geometry.approach_length_m
        self.loops: dict[str, LoopDetector] = {}
        self.trap_downstream: dict[str, LoopDetector] = {}
        self.speed_samples: dict[str, deque[SpeedSample]] = {}
        self._trap_pending: dict[str, deque[float]] = {}
        for approach in Approach.ordered():
            for lane in range(INBOUND_LANES):
                base = f"{approach.value}.{lane}"
                self._add(LoopDetector(f"{base}.stop", approach, lane, LoopRole.STOP, L - 3.0, L - 1.0))
                adv_start = L - ADVANCE_SETBACK_M
                self._add(
                    LoopDetector(
                        f"{base}.advance", approach, lane, LoopRole.ADVANCE, adv_start, adv_start + 2
                    )
                )
                trap_start = adv_start + SPEED_TRAP_SPACING_M
                self.trap_downstream[f"{base}.advance"] = LoopDetector(
                    f"{base}.advance.b",
                    approach,
                    lane,
                    LoopRole.ADVANCE,
                    trap_start,
                    trap_start + LOOP_LENGTH_M,
                )
                self.speed_samples[f"{base}.advance"] = deque(maxlen=200)
                self._trap_pending[f"{base}.advance"] = deque(maxlen=8)
                entry_start = ENTRY_SETBACK_FROM_START_M
                self._add(
                    LoopDetector(
                        f"{base}.entry",
                        approach,
                        lane,
                        LoopRole.ENTRY,
                        entry_start,
                        entry_start + LOOP_LENGTH_M,
                    )
                )
        self._false_rate_per_tick = profile.false_actuations_per_hour / 3600.0 * engine.dt

    def _add(self, loop: LoopDetector) -> None:
        self.loops[loop.detector_id] = loop

    def get(self, approach: Approach, lane: int, kind: DetectorKind) -> LoopDetector:
        return self.loops[f"{approach.value}.{lane}.{kind}"]

    # ------------------------------------------------------------------ sampling
    def sample(self) -> None:
        t = self._engine.time
        self._apply_faults(t)
        f = self._engine.fleet
        idx = f.indices
        idx = idx[f.stage[idx] == Stage.INBOUND]
        link = f.approach[idx].astype(np.intp) * INBOUND_LANES + f.lane[idx]
        front = f.s[idx]
        rear = front - f.length[idx]
        miss = f.miss_mask[idx]

        for loop in self.loops.values():
            on_lane = link == loop.approach.ordinal * INBOUND_LANES + loop.lane
            truth = self._overlaps(on_lane, front, rear, miss, loop.start_m, loop.end_m, loop.role)
            self._update(loop, truth, t)
            if loop.role is LoopRole.ADVANCE:
                self._update_trap(loop, on_lane, front, rear, miss, t)

    @staticmethod
    def _overlaps(
        on_lane: npt.NDArray[np.bool_],
        front: npt.NDArray[np.float64],
        rear: npt.NDArray[np.float64],
        miss: npt.NDArray[np.int8],
        start: float,
        end: float,
        role: LoopRole,
    ) -> bool:
        seen = (miss >> role.value) & 1 == 0
        return bool(np.any(on_lane & seen & (front >= start) & (rear <= end)))

    def _update(self, loop: LoopDetector, truth: bool, t: float) -> None:
        present = self._measured(loop, truth, t)
        rising = present and not loop.present
        if rising:
            loop.rising_edges += 1
            loop.last_on_time = t
            loop.on_since = t
        elif not present:
            loop.on_since = None
        if present:
            loop.last_present_time = t
        loop.present = present
        loop.history.append((t, present, rising))

    def _measured(self, loop: LoopDetector, truth: bool, t: float) -> bool:
        match loop.fault:
            case "stuck_on":
                return True
            case "dead":
                return False
            case "chatter":
                loop._chatter_phase += self._rng.uniform(0.2, 1.0)
                return truth or (int(loop._chatter_phase) % 2 == 1)
            case _:
                pass
        if self._rng.random() < self._false_rate_per_tick:
            loop._false_until = t + float(self._rng.uniform(0.1, 0.4))
        return truth or t < loop._false_until

    def _update_trap(
        self,
        loop: LoopDetector,
        on_lane: npt.NDArray[np.bool_],
        front: npt.NDArray[np.float64],
        rear: npt.NDArray[np.float64],
        miss: npt.NDArray[np.int8],
        t: float,
    ) -> None:
        """Speed = spacing / (time B turns on - time A turned on), FIFO-matched like real traps."""
        b = self.trap_downstream[loop.detector_id]
        truth_b = self._overlaps(on_lane, front, rear, miss, b.start_m, b.end_m, LoopRole.ADVANCE)
        present_b = truth_b if loop.fault is None else loop.present
        pending = self._trap_pending[loop.detector_id]
        if loop.present and loop.last_on_time == t:
            pending.append(t)
        if present_b and not b.present:
            while pending and t - pending[0] > 6.0:
                pending.popleft()  # stale: vehicle stopped between loops or was missed
            if pending:
                dt = t - pending.popleft()
                if dt > 0:
                    speed = SPEED_TRAP_SPACING_M / dt
                    if 0.5 < speed < 40.0:  # plausibility filter
                        self.speed_samples[loop.detector_id].append(SpeedSample(t, speed))
        b.present = present_b

    def _apply_faults(self, t: float) -> None:
        active: dict[str, str] = {}
        for fault in self._profile.faults:
            if fault.start_s <= t < fault.start_s + fault.duration_s:
                active[_fault_id(fault)] = fault.mode
        for loop_id, loop in self.loops.items():
            loop.fault = active.get(loop_id)


def _fault_id(fault: FaultEvent) -> str:
    return f"{fault.approach.value}.{fault.lane}.{fault.detector}"


__all__ = ["ADVANCE_SETBACK_M", "DetectorArray", "LoopDetector", "LoopRole", "SpeedSample"]
