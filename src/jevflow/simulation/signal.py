"""The signal controller: the hardware-side state machine that keeps the intersection safe.

Control strategies (including Jev) never set lights directly. They send commands, and this
controller accepts them only when they are safe, the same way a field controller enforces its
timing and a conflict monitor guards the outputs. It owns:

* the GREEN → YELLOW → ALL_RED → next GREEN sequence,
* minimum green, including the longer minimum needed to serve a pedestrian call,
* max-out when a conflicting call has waited through ``max_green``,
* pedestrian WALK / flashing DON'T WALK,
* emergency-vehicle preemption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jevflow.domain.enums import Approach, ControlAction, Interval, Phase, SignalColor
from jevflow.domain.timing import SignalTiming


class IllegalCommandError(RuntimeError):
    """A strategy asked for something the controller must refuse."""


class PedSignal(StrEnum):
    WALK = "walk"
    FLASHING_DONT_WALK = "flash"
    DONT_WALK = "dont_walk"


@dataclass(slots=True)
class PhaseChange:
    """Emitted when a new green starts; useful for logs and metrics."""

    time: float
    phase: Phase
    previous_green_s: float
    reason: str


class SignalController:
    def __init__(self, timing: SignalTiming, initial_phase: Phase = Phase.NORTH_SOUTH) -> None:
        self.timing = timing
        self.phase = initial_phase
        self.interval = Interval.GREEN
        self.time = 0.0
        self.interval_elapsed = 0.0
        self.green_elapsed = 0.0
        self.hold_until = 0.0
        self.preempted_by: Approach | None = None
        self.ped_calls: dict[Phase, bool] = dict.fromkeys(Phase, False)
        self._walk_phase: Phase | None = None
        self._walk_end_at = timing.walk_s
        self._min_green = timing.min_green_s
        self._green_ended_at: dict[Approach, float] = dict.fromkeys(Approach, 0.0)
        self._termination_reason = ""
        self.phase_changes: list[PhaseChange] = []

    # ------------------------------------------------------------------ queries
    @property
    def min_green_effective(self) -> float:
        return self._min_green

    @property
    def extension_remaining(self) -> float:
        if self.interval is not Interval.GREEN:
            return 0.0
        return max(0.0, self.hold_until - self.green_elapsed)

    def color(self, approach: Approach) -> SignalColor:
        if approach.phase is not self.phase or self.interval is Interval.ALL_RED:
            return SignalColor.RED
        return SignalColor.GREEN if self.interval is Interval.GREEN else SignalColor.YELLOW

    def colors(self) -> dict[Approach, SignalColor]:
        return {a: self.color(a) for a in Approach}

    def time_since_last_green(self, approach: Approach) -> float:
        if self.color(approach) is SignalColor.GREEN:
            return 0.0
        return self.time - self._green_ended_at[approach]

    def ped_signal(self, leg: Approach) -> PedSignal:
        """Pedestrian head for the crosswalk across ``leg`` (served by the *other* phase)."""
        serving = leg.phase.other
        if (
            self._walk_phase is not serving
            or self.phase is not serving
            or self.interval is not Interval.GREEN
        ):
            return PedSignal.DONT_WALK
        if self.green_elapsed < self._walk_end_at:
            return PedSignal.WALK
        if self.green_elapsed < self._ped_clearance_end():
            return PedSignal.FLASHING_DONT_WALK
        return PedSignal.DONT_WALK

    def can_terminate(self) -> bool:
        if self.interval is not Interval.GREEN:
            return False
        if self.preempted_by is not None and self.preempted_by.phase is self.phase:
            return False
        return self.green_elapsed >= self._min_green

    def can_extend(self) -> bool:
        return (
            self.interval is Interval.GREEN
            and self.green_elapsed + self.timing.extension_s <= self.timing.max_green_s
        )

    # ------------------------------------------------------------------ commands
    def call_pedestrian(self, leg: Approach) -> None:
        # A push during WALK is served now; any later push waits for the next service.
        if self.ped_signal(leg) is not PedSignal.WALK:
            self.ped_calls[leg.phase.other] = True

    def set_preemption(self, approach: Approach | None) -> None:
        self.preempted_by = approach

    def command(self, action: ControlAction) -> None:
        match action:
            case ControlAction.KEEP_CURRENT_PHASE:
                return
            case ControlAction.EXTEND_CURRENT_GREEN:
                if not self.can_extend():
                    raise IllegalCommandError("extension would exceed maximum green")
                self.hold_until = self.green_elapsed + self.timing.extension_s
            case ControlAction.SWITCH_PHASE:
                if not self.can_terminate():
                    raise IllegalCommandError("phase cannot terminate yet (min green, walk or preemption)")
                self._start_yellow("strategy")

    # ------------------------------------------------------------------ time
    def step(self, dt: float, conflicting_call: bool) -> None:
        self.time += dt
        self.interval_elapsed += dt
        match self.interval:
            case Interval.GREEN:
                self.green_elapsed += dt
                self._step_green(conflicting_call)
            case Interval.YELLOW if self.interval_elapsed >= self.timing.yellow_s:
                self.interval = Interval.ALL_RED
                self.interval_elapsed = 0.0
            case Interval.ALL_RED if self.interval_elapsed >= self.timing.all_red_s:
                self._start_green(self.phase.other)
            case _:
                pass

    def _step_green(self, conflicting_call: bool) -> None:
        preempt = self.preempted_by
        if preempt is not None and preempt.phase is not self.phase:
            # Preemption truncates WALK but never the pedestrian clearance.
            self._walk_end_at = min(self._walk_end_at, self.green_elapsed)
            ready = max(self.timing.preempt_min_green_s, self._ped_clearance_end())
            if self.green_elapsed >= ready:
                self._start_yellow("preemption")
            return
        if preempt is not None:
            return  # hold green for the emergency vehicle
        if (
            conflicting_call
            and self.green_elapsed >= self.timing.max_green_s
            and self.green_elapsed >= self._min_green
        ):
            self._start_yellow("max-out")

    def _ped_clearance_end(self) -> float:
        if self._walk_phase is not self.phase:
            return 0.0
        return self._walk_end_at + self.timing.ped_clearance_s

    def _start_yellow(self, reason: str) -> None:
        self.interval = Interval.YELLOW
        self.interval_elapsed = 0.0
        self.hold_until = 0.0
        self._termination_reason = reason
        for approach in self.phase.approaches:
            self._green_ended_at[approach] = self.time

    def _start_green(self, phase: Phase) -> None:
        self.phase_changes.append(PhaseChange(self.time, phase, self.green_elapsed, self._termination_reason))
        self.phase = phase
        self.interval = Interval.GREEN
        self.interval_elapsed = 0.0
        self.green_elapsed = 0.0
        self.hold_until = 0.0
        self._walk_end_at = self.timing.walk_s
        self._min_green = self.timing.min_green_s
        if self.ped_calls[phase]:
            self.ped_calls[phase] = False
            self._walk_phase = phase
            self._min_green = max(self._min_green, self.timing.ped_min_green_s)
        else:
            self._walk_phase = None
