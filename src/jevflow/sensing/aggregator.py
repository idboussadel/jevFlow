"""Traffic-state aggregator: loop actuations in, controller-grade measurements out.

This is where "the simulator knows everything" becomes "the controller knows what its
detectors tell it". The estimators are the standard field techniques:

* **Queue by input–output counting** between the upstream entry loop and the stop-line loop,
  minus vehicles still approaching the back of the queue and vehicles released by the green
  start-up wave. It drifts with every missed count, so it is re-anchored whenever the stop-line
  loop shows the queue has fully discharged during green.
* **Spillback**: the entry loop standing occupied, or the lane counted as full.
* **Delay by FIFO re-identification**: each input actuation is matched to the next stop-line
  actuation, and the delay is travel time minus free-flow time.
* **Speed**: speed-trap harmonic mean, with the single-loop occupancy estimator as a fallback.
* **Detector health**: stuck-on, dead and chattering loops are flagged, and the lane falls back
  to a model based on historical volumes, the way agencies degrade gracefully in the field.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from jevflow.domain.enums import Approach, Phase, SignalColor
from jevflow.domain.geometry import INBOUND_LANES, LEFT_LANE
from jevflow.domain.observation import (
    ApproachMeasurement,
    DetectorHealth,
    SignalStatus,
    TrafficObservation,
)
from jevflow.sensing.detectors import (
    ADVANCE_SETBACK_M,
    ENTRY_SETBACK_FROM_START_M,
    DetectorArray,
    LoopDetector,
)
from jevflow.simulation.engine import SimulationEngine

WINDOW_S = 30.0
JAM_SPACING_M = 7.0
EFFECTIVE_LOOP_LENGTH_M = 6.7  # mean vehicle length + loop length, for single-loop speed
SATURATION_RATE_VPS = 0.5  # ~1800 veh/h/lane, for fallbacks
STUCK_ON_S = 120.0
DEAD_S = 150.0
CHATTER_EDGES_PER_WINDOW = 24
SPILLBACK_ON_S = 4.0
DRIFT_RELAXATION = 0.03  # per 100 ms scan, ~26% per second
PREEMPT_DETECTION_M = 300.0


@dataclass(slots=True)
class LaneEstimator:
    """Per-lane queue and delay estimation from three loops.

    Input–output counting between the entry loop and the stop-line loop gives the number of
    vehicles on the lane. Vehicles still travelling towards the back of the queue, and vehicles
    already released by the green discharge wave, are subtracted to get the *standing* queue.
    The count drifts with every missed or false actuation; it is re-anchored each time the
    stop-line loop shows the queue has fully discharged.
    """

    stop: LoopDetector
    advance: LoopDetector
    entry: LoopDetector
    free_flow_ms: float
    entry_to_stop_m: float
    capacity_total: float
    on_lane: float = 0.0
    queue: float = 0.0
    fifo: deque[float] = field(default_factory=deque)
    delays: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=400))
    inputs_hist: deque[float] = field(default_factory=lambda: deque(maxlen=900))
    departures_hist: deque[float] = field(default_factory=lambda: deque(maxlen=900))
    faults: dict[str, str] = field(default_factory=dict)
    discharge_gap_s: float = 3.0  # stop-line gap that means "queue cleared" (longer for permissive lefts)
    red_started: float | None = None
    anchor_queue: float = 0.0  # standing queue when the current interval began
    departed_this_green: int = 0

    @property
    def input_loop(self) -> LoopDetector:
        """Entry loop normally; the advance loop (shorter coverage) if the entry loop is faulty."""
        return self.advance if "entry" in self.faults else self.entry

    @property
    def travel_time(self) -> float:
        return self._input_distance / self.free_flow_ms

    @property
    def _input_distance(self) -> float:
        return ADVANCE_SETBACK_M if "entry" in self.faults else self.entry_to_stop_m

    @property
    def degraded(self) -> bool:
        return "stop" in self.faults or ("entry" in self.faults and "advance" in self.faults)

    def update(self, t: float, color: SignalColor, green_elapsed: float) -> None:
        loop_in = self.input_loop
        in_rise = bool(loop_in.history and loop_in.history[-1][2])
        stop_rise = bool(self.stop.history and self.stop.history[-1][2]) and "stop" not in self.faults

        if in_rise:
            self.inputs_hist.append(t)
            self.fifo.append(t)
        if stop_rise:
            self.departures_hist.append(t)
            if self.fifo:
                arrived = self.fifo.popleft()
                self.delays.append((t, max(0.0, t - arrived - self.travel_time)))

        if color is SignalColor.GREEN:
            if self.red_started is not None:
                self.anchor_queue = self.queue
                self.departed_this_green = 0
            self.red_started = None
            self.departed_this_green += stop_rise
        elif self.red_started is None:
            self.red_started = t

        if self.degraded:
            self._model_based(t, color, green_elapsed)
        else:
            self._count_based(t, in_rise, stop_rise, color, green_elapsed)

    def _count_based(
        self, t: float, in_rise: bool, stop_rise: bool, color: SignalColor, green_elapsed: float
    ) -> None:
        self.on_lane = min(self.capacity_total, max(0.0, self.on_lane + in_rise - stop_rise))

        stop_gap = t - self.stop.last_present_time
        advance_gap = t - self.advance.last_present_time
        cleared = stop_gap > self.discharge_gap_s and advance_gap > self.discharge_gap_s
        if color is SignalColor.GREEN and cleared and green_elapsed > 4.0:
            # The queue has discharged. Relax the count towards the vehicles still en route. A
            # gentle correction absorbs detector drift without discarding real upstream queues.
            en_route = float(self._recent(self.inputs_hist, t, 1.5 * self.travel_time))
            if self.on_lane > en_route:
                self.on_lane += (en_route - self.on_lane) * DRIFT_RELAXATION
                while len(self.fifo) > max(self.on_lane, en_route) + 0.5:
                    self.fifo.popleft()
            self.anchor_queue = 0.0

        # Vehicles that entered recently are still driving towards the back of the queue.
        tail_distance = self.queue * JAM_SPACING_M
        time_to_tail = max(0.0, self._input_distance - tail_distance) / self.free_flow_ms
        approaching = self._recent(self.inputs_hist, t, time_to_tail)

        released = 0.0
        if color is SignalColor.GREEN:
            # The start-up wave frees roughly one queued vehicle per 1.1 s after ~1 s of reaction.
            started = max(0.0, green_elapsed - 1.0) / 1.1
            released = max(0.0, min(started, self.anchor_queue) - self.departed_this_green)

        self.queue = max(0.0, self.on_lane - approaching - released)
        if self.spillback(t):
            self.queue = max(self.queue, self.capacity_total)

    def _model_based(self, t: float, color: SignalColor, green_elapsed: float) -> None:
        """Historical-rate model used when the lane cannot be counted reliably."""
        history = self.departures_hist if "stop" not in self.faults else self.inputs_hist
        rate = max(self._recent(history, t, 300.0) / 300.0, 0.03)
        if color is SignalColor.GREEN:
            self.queue = max(0.0, self.anchor_queue + (rate - SATURATION_RATE_VPS) * green_elapsed)
        else:
            waited = t - (self.red_started or t)
            self.queue = min(self.capacity_total, self.anchor_queue + rate * waited)
        self.on_lane = self.queue

    def spillback(self, t: float) -> bool:
        if "entry" in self.faults:
            return False
        standing = self.entry.on_since is not None and t - self.entry.on_since >= SPILLBACK_ON_S
        return standing or self.on_lane >= 0.92 * self.capacity_total

    def average_delay(self, t: float, window: float = 90.0) -> float:
        done = [d for (ts, d) in self.delays if t - ts <= window]
        waiting = [max(0.0, t - a - self.travel_time) for a in self.fifo]
        samples = done + waiting
        return sum(samples) / len(samples) if samples else 0.0

    @staticmethod
    def _recent(times: deque[float], t: float, window: float) -> int:
        n = 0
        for ts in reversed(times):
            if t - ts > window:
                break
            n += 1
        return n


class TrafficStateAggregator:
    def __init__(self, engine: SimulationEngine, detectors: DetectorArray) -> None:
        self._engine = engine
        self._detectors = detectors
        geometry = engine.geometry
        v_ff = geometry.speed_limit_ms
        entry_to_stop = geometry.approach_length_m - ENTRY_SETBACK_FROM_START_M
        self.lanes: dict[tuple[Approach, int], LaneEstimator] = {
            (a, lane): LaneEstimator(
                stop=detectors.get(a, lane, "stop"),
                advance=detectors.get(a, lane, "advance"),
                entry=detectors.get(a, lane, "entry"),
                free_flow_ms=v_ff,
                entry_to_stop_m=entry_to_stop,
                capacity_total=entry_to_stop / JAM_SPACING_M,
                discharge_gap_s=6.0 if lane == LEFT_LANE else 3.0,
            )
            for a in Approach
            for lane in range(INBOUND_LANES)
        }
        self._flags: dict[str, str] = {}

    # ------------------------------------------------------------------ per tick
    def update(self) -> None:
        engine = self._engine
        t = engine.time
        if round(t * 10) % 10 == 0:
            self._diagnose(t)
        signal = engine.signal
        for (approach, _lane), est in self.lanes.items():
            est.faults = {
                kind: self._flags[loop.detector_id]
                for kind, loop in (("stop", est.stop), ("advance", est.advance), ("entry", est.entry))
                if loop.detector_id in self._flags
            }
            est.update(t, signal.color(approach), signal.green_elapsed)

    def _diagnose(self, t: float) -> None:
        """Flag loops whose behaviour is physically implausible."""
        flags: dict[str, str] = {}
        for est in self.lanes.values():
            loops = (est.stop, est.advance, est.entry)
            for loop in loops:
                if loop.on_since is not None and t - loop.on_since >= STUCK_ON_S:
                    # A standing queue can hold a loop on for minutes; a stuck loop stays on
                    # while the stop line keeps discharging vehicles past it.
                    moved = sum(1 for (ts, _, r) in est.stop.history if r and ts >= loop.on_since)
                    if loop is est.stop or moved >= 4:
                        flags[loop.detector_id] = "stuck_on"
                        continue
                edges = sum(1 for (ts, _, r) in loop.history if r and t - ts <= WINDOW_S)
                if edges >= CHATTER_EDGES_PER_WINDOW:
                    flags[loop.detector_id] = "chattering"
                    continue
                silent_for = t - loop.last_on_time
                if silent_for >= DEAD_S and not loop.present:
                    partner = est.advance if loop is est.stop else est.stop
                    if partner.last_on_time > loop.last_on_time + 60.0 and partner.rising_edges > 8:
                        flags[loop.detector_id] = "no_activity"
        self._flags = flags

    # ------------------------------------------------------------------ snapshot
    def observe(self) -> TrafficObservation:
        engine = self._engine
        t = engine.time
        signal = engine.signal
        approaches = {a: self._measure(a, t) for a in Approach}
        status = SignalStatus(
            current_phase=signal.phase,
            interval=signal.interval,
            colors=signal.colors(),
            interval_elapsed_seconds=round(signal.interval_elapsed, 1),
            phase_elapsed_seconds=round(signal.green_elapsed, 1),
            minimum_green_seconds=signal.min_green_effective,
            maximum_green_seconds=signal.timing.max_green_s,
            yellow_seconds=signal.timing.yellow_s,
            all_red_seconds=signal.timing.all_red_s,
            extension_remaining_seconds=round(signal.extension_remaining, 1),
            time_since_last_green={a: round(signal.time_since_last_green(a), 1) for a in Approach},
            preempted_by=signal.preempted_by,
        )
        ev = self.emergency_detection()
        return TrafficObservation(
            time_seconds=round(t, 1),
            signal=status,
            approaches=approaches,
            pedestrian_calls={p: signal.ped_calls[p] for p in Phase},
            emergency_vehicle_approach=ev[0] if ev else None,
            emergency_vehicle_distance_m=round(ev[1], 0) if ev else None,
        )

    def emergency_detection(self) -> tuple[Approach, float] | None:
        """Priority-vehicle transponder detection (optical/GPS preemption emitters)."""
        detected = [(a, d) for a, d in self._engine.emergency_vehicles() if d <= PREEMPT_DETECTION_M]
        return min(detected, key=lambda x: x[1]) if detected else None

    def has_conflicting_call(self) -> bool:
        """Vehicle or pedestrian demand waiting on the phase that is not green."""
        signal = self._engine.signal
        other = signal.phase.other
        if signal.ped_calls[other]:
            return True
        return any(
            est.queue >= 0.5 or est.stop.present
            for (approach, _), est in self.lanes.items()
            if approach.phase is other
        )

    def detector_states(self) -> list[tuple[str, bool, str | None]]:
        return [
            (d.detector_id, d.present, self._flags.get(d.detector_id)) for d in self._detectors.loops.values()
        ]

    def estimated_queues(self) -> dict[Approach, float]:
        return {a: sum(self.lanes[(a, lane)].queue for lane in range(INBOUND_LANES)) for a in Approach}

    def _measure(self, approach: Approach, t: float) -> ApproachMeasurement:
        lanes = [self.lanes[(approach, lane)] for lane in range(INBOUND_LANES)]
        adv = [e.advance for e in lanes]
        stops = [e.stop for e in lanes]
        volume = sum(_edges(loop, t) for loop in adv)
        departures = sum(_edges(loop, t) for loop in stops)
        occupancy = sum(_occupancy(loop, t) for loop in adv) / len(adv)
        stop_occ = sum(_occupancy(loop, t) for loop in stops) / len(stops)

        samples = [
            s.speed_ms
            for loop in adv
            for s in self._detectors.speed_samples[loop.detector_id]
            if t - s.time <= WINDOW_S
        ]
        if samples:
            speed = len(samples) / sum(1.0 / v for v in samples)  # harmonic mean ≈ space-mean speed
        elif occupancy > 2.0:
            flow = volume / WINDOW_S / len(adv)
            speed = flow * EFFECTIVE_LOOP_LENGTH_M / (occupancy / 100.0)
        else:
            speed = self._engine.geometry.speed_limit_ms
        faults = tuple(
            sorted({loop_id for loop_id in self._flags if loop_id.startswith(f"{approach.value}.")})
        )
        last_actuation = max(loop.last_on_time for e in lanes for loop in (e.stop, e.advance))
        return ApproachMeasurement(
            volume_vehicles_30s=volume,
            departures_30s=departures,
            occupancy_percent=round(min(occupancy, 100.0), 1),
            stop_line_occupancy_percent=round(min(stop_occ, 100.0), 1),
            average_speed_kmh=round(min(speed * 3.6, 90.0), 1),
            estimated_queue_vehicles=round(sum(e.queue for e in lanes), 1),
            estimated_left_turn_queue_vehicles=round(self.lanes[(approach, LEFT_LANE)].queue, 1),
            average_delay_seconds=round(
                max((e.average_delay(t) for e in lanes), default=0.0)
                if any(e.queue > 0 for e in lanes)
                else sum(e.average_delay(t) for e in lanes) / len(lanes),
                1,
            ),
            queue_spillback_detected=any(e.spillback(t) for e in lanes),
            detector_health=DetectorHealth.DEGRADED if faults else DetectorHealth.OK,
            faulty_detectors=faults,
            seconds_since_last_actuation=round(max(0.0, min(t - last_actuation, 999.0)), 1),
        )


def _edges(loop: LoopDetector, t: float) -> int:
    return sum(1 for (ts, _, rising) in loop.history if rising and t - ts <= WINDOW_S)


def _occupancy(loop: LoopDetector, t: float) -> float:
    samples = [present for (ts, present, _) in loop.history if t - ts <= WINDOW_S]
    return 100.0 * sum(samples) / len(samples) if samples else 0.0
