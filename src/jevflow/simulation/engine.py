"""Microscopic simulation engine: the *ground truth* the rest of the system cannot see directly.

Each 100 ms tick:

1. the signal controller advances its intervals;
2. pedestrians arrive, press buttons, cross on WALK;
3. scheduled arrivals enter their lane, or wait in an off-map *vertical queue* when the lane
   has spilled back to its entry;
4. every vehicle finds its leader (a vectorised sort by link and position). Heads of lanes also
   see virtual obstacles: the stop line on red, on yellow when the driver chose to stop, when
   a permissive left has no acceptable gap, or when a turner must yield to pedestrians;
5. IDM accelerations are computed, drivers who start from standstill wait their reaction time
   (this is what produces realistic start-up lost time), and positions are updated ballistically;
6. stage transitions record red-light violations, control delay and stops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jevflow.domain.enums import Approach, Interval, Movement, SignalColor, VehicleClass
from jevflow.domain.geometry import (
    INBOUND_LANES,
    LEFT_LANE,
    THROUGH_RIGHT_LANE,
    IntersectionGeometry,
    exit_approach,
    outbound_lane_for_movement,
)
from jevflow.domain.timing import SignalTiming
from jevflow.simulation.demand import Arrival, generate_arrivals, generate_pedestrians
from jevflow.simulation.fleet import Fleet, Stage, YellowDecision
from jevflow.simulation.idm import ballistic_update, idm_acceleration
from jevflow.simulation.pedestrians import PedestrianModel
from jevflow.simulation.rng import RandomStreams
from jevflow.simulation.scenario import Scenario
from jevflow.simulation.signal import SignalController

F = npt.NDArray[np.float64]
I = npt.NDArray[np.intp]  # noqa: E741

DT = 0.1
QUEUE_SPEED_MS = 2.0  # below ~7 km/h a vehicle counts as queued (HCM uses 5 mph)
STOP_LINE_SETBACK_M = 1.0
TURN_SPEED_MS = {Movement.LEFT: 7.0, Movement.RIGHT: 4.8}
YIELD_LOOKAHEAD_M = 25.0
SNEAKER_ZONE_M = 10.0  # head of the left queue and the vehicle directly behind it

_GREEN, _YELLOW, _RED = 0, 1, 2
_COLOR_CODE = {SignalColor.GREEN: _GREEN, SignalColor.YELLOW: _YELLOW, SignalColor.RED: _RED}
_N_LINKS = len(Approach) * INBOUND_LANES


@dataclass(frozen=True, slots=True)
class TripRecord:
    approach: int
    movement: int
    vclass: int
    delay: float
    stops: int
    completed_at: float


@dataclass(frozen=True, slots=True)
class VehiclePoses:
    vid: npt.NDArray[np.int64]
    x: F
    y: F
    heading: F
    cls: npt.NDArray[np.int8]
    braking: npt.NDArray[np.bool_]
    turn: npt.NDArray[np.int8]  # -1 left, 0 none, 1 right (for indicators)
    speed: F


def resolve_timing(scenario: Scenario) -> SignalTiming:
    """Scenario override, or timing engineered from geometry with the ITE formulas."""
    if scenario.timing is not None:
        return scenario.timing
    g = IntersectionGeometry(approach_length_m=scenario.approach_length_m)
    return SignalTiming.engineered(g.speed_limit_ms, g.crossing_distance_m + 2.0, g.crossing_distance_m)


class SimulationEngine:
    def __init__(self, scenario: Scenario, seed: int, timing: SignalTiming | None = None) -> None:
        self.scenario = scenario
        self.seed = seed
        self.geometry = IntersectionGeometry(approach_length_m=scenario.approach_length_m)
        g = self.geometry
        self.timing = timing or resolve_timing(scenario)
        self.signal = SignalController(self.timing)
        self.streams = RandomStreams(seed)
        self._noise = self.streams.get("acceleration-noise")
        self._pending: deque[Arrival] = deque(generate_arrivals(scenario, self.streams, g.speed_limit_ms))
        self.entry_queues: list[deque[Arrival]] = [deque() for _ in range(_N_LINKS)]
        self.pedestrians = PedestrianModel(generate_pedestrians(scenario, self.streams), g)
        self.fleet = Fleet()
        self.time = 0.0
        self.trips: list[TripRecord] = []
        self.red_light_violations = 0
        self.departures = np.zeros(len(Approach), dtype=np.int64)
        self._L = g.approach_length_m
        self._exit_len = g.exit_length_m
        self._travel = np.array([a.travel_direction for a in Approach.ordered()])
        self._paths = [g.path(a, m) for a in Approach.ordered() for m in Movement]

    # ================================================================== public API
    @property
    def dt(self) -> float:
        return DT

    @property
    def finished(self) -> bool:
        return self.time >= self.scenario.duration_s

    def step(self, conflicting_call: bool) -> None:
        self.signal.step(DT, conflicting_call)
        self.time = self.signal.time
        self.pedestrians.step(self.time, DT, self.signal)
        self._release_arrivals()
        self._insert_vehicles()
        idx = self.fleet.indices
        if idx.size == 0:
            return
        colors = self._color_codes()
        self._update_yellow_decisions(idx, colors)
        acc = self._accelerations(idx, colors)
        self._integrate(idx, acc)
        self._transitions(idx, colors)

    # ------------------------------------------------------------------ truth queries
    def vertical_queue(self) -> dict[Approach, int]:
        return {
            a: sum(len(self.entry_queues[a.ordinal * INBOUND_LANES + lane]) for lane in range(INBOUND_LANES))
            for a in Approach
        }

    def true_queues(self) -> dict[Approach, int]:
        f = self.fleet
        idx = f.indices
        queued = idx[(f.stage[idx] == Stage.INBOUND) & (f.v[idx] < QUEUE_SPEED_MS)]
        counts = np.bincount(f.approach[queued], minlength=len(Approach))
        vertical = self.vertical_queue()
        return {a: int(counts[a.ordinal]) + vertical[a] for a in Approach}

    def current_waits(self) -> dict[Approach, float]:
        """Longest accumulated stopped time among vehicles still on each approach."""
        f = self.fleet
        idx = f.indices
        inbound = idx[f.stage[idx] == Stage.INBOUND]
        out = dict.fromkeys(Approach, 0.0)
        for i in inbound:
            a = Approach.from_index(int(f.approach[i]))
            out[a] = max(out[a], float(f.wait_total[i]))
        for a in Approach:
            for lane in range(INBOUND_LANES):
                q = self.entry_queues[a.ordinal * INBOUND_LANES + lane]
                if q:
                    out[a] = max(out[a], self.time - q[0].time)
        return out

    def emergency_vehicles(self) -> list[tuple[Approach, float]]:
        """(approach, distance to stop line) for every inbound emergency vehicle."""
        f = self.fleet
        idx = f.indices
        ev = idx[(f.cls[idx] == VehicleClass.EMERGENCY) & (f.stage[idx] == Stage.INBOUND)]
        return [(Approach.from_index(int(f.approach[i])), float(self._L - f.s[i])) for i in ev]

    def unfinished_delay(self) -> list[float]:
        """Delay already accrued by vehicles that have not cleared the junction yet."""
        f = self.fleet
        idx = f.indices
        inbound = idx[f.stage[idx] == Stage.INBOUND]
        accrued = (self.time - f.arrival_time[inbound]) - f.s[inbound] / f.v0[inbound]
        waiting = [self.time - arr.time for q in self.entry_queues for arr in q]
        return [*np.maximum(accrued, 0.0).tolist(), *waiting]

    def poses(self) -> VehiclePoses:
        f = self.fleet
        idx = f.indices
        n = idx.size
        x = np.empty(n)
        y = np.empty(n)
        heading = np.empty(n)
        stage = f.stage[idx]
        s = f.s[idx]
        ap = f.approach[idx].astype(np.intp)
        g = self.geometry

        m = stage == Stage.INBOUND
        if m.any():
            dx, dy = self._travel[ap[m], 0], self._travel[ap[m], 1]
            back = g.stop_line_offset_m + self._L - s[m]
            off = (f.lane[idx][m] + 0.5) * g.lane_width_m
            x[m] = -dx * back + dy * off
            y[m] = -dy * back - dx * off
            heading[m] = np.arctan2(dy, dx)

        m = stage == Stage.BOX
        if m.any():
            pid = ap * 3 + f.movement[idx].astype(np.intp)
            u = s - self._L
            for p in np.unique(pid[m]):
                sel = m & (pid == p)
                path = self._paths[int(p)]
                x[sel] = np.interp(u[sel], path.s, path.x)
                y[sel] = np.interp(u[sel], path.s, path.y)
                heading[sel] = np.interp(u[sel], path.s, path.heading)

        m = stage == Stage.OUTBOUND
        if m.any():
            ol = f.out_link[idx][m].astype(np.intp)
            leg = ol // INBOUND_LANES
            ox, oy = -self._travel[leg, 0], -self._travel[leg, 1]
            dist = g.stop_line_offset_m + (s[m] - self._L - f.path_len[idx][m])
            off = (ol % INBOUND_LANES + 0.5) * g.lane_width_m
            x[m] = ox * dist + oy * off
            y[m] = oy * dist - ox * off
            heading[m] = np.arctan2(oy, ox)

        mv = f.movement[idx]
        signalling = (stage == Stage.BOX) | ((stage == Stage.INBOUND) & (self._L - s < 70.0))
        turn = np.where(
            signalling & (mv == Movement.LEFT), -1, np.where(signalling & (mv == Movement.RIGHT), 1, 0)
        )
        return VehiclePoses(
            vid=f.vid[idx].copy(),
            x=x,
            y=y,
            heading=heading,
            cls=f.cls[idx].copy(),
            braking=(f.acc[idx] < -0.6) | ((f.v[idx] < 0.3) & (stage == Stage.INBOUND)),
            turn=turn.astype(np.int8),
            speed=f.v[idx].copy(),
        )

    # ================================================================== internals
    def _color_codes(self) -> npt.NDArray[np.int8]:
        return np.array([_COLOR_CODE[self.signal.color(a)] for a in Approach.ordered()], dtype=np.int8)

    def _release_arrivals(self) -> None:
        while self._pending and self._pending[0].time <= self.time:
            arrival = self._pending.popleft()
            self.entry_queues[arrival.approach.ordinal * INBOUND_LANES + arrival.lane].append(arrival)

    def _insert_vehicles(self) -> None:
        f = self.fleet
        idx = f.indices
        inbound = idx[f.stage[idx] == Stage.INBOUND]
        link = f.approach[inbound].astype(np.intp) * INBOUND_LANES + f.lane[inbound]
        for lane_id, queue in enumerate(self.entry_queues):
            if not queue:
                continue
            arrival = queue[0]
            d = arrival.driver
            on_lane = inbound[link == lane_id]
            speed = d.desired_speed * 0.95
            if on_lane.size:
                tail = on_lane[np.argmin(f.s[on_lane])]
                gap = f.s[tail] - f.length[tail]
                if gap < d.min_gap + 1.0:
                    continue  # lane has spilled back to its entry
                speed = min(speed, max(0.0, (gap - d.min_gap) / d.time_gap), f.v[tail] + 2.0)
            queue.popleft()
            self._spawn(arrival, speed)

    def _spawn(self, arrival: Arrival, speed: float) -> None:
        movement = arrival.movement
        path = self.geometry.path(arrival.approach, movement)
        leg = exit_approach(arrival.approach, movement)
        out_link = leg.ordinal * INBOUND_LANES + outbound_lane_for_movement(movement)
        turn_speed = TURN_SPEED_MS.get(movement, arrival.driver.desired_speed)
        v0 = arrival.driver.desired_speed
        ff_time = self._L / v0 + path.length / min(v0, turn_speed)
        self.fleet.add(
            arrival,
            speed=speed,
            out_link=out_link,
            path_len=path.length,
            turn_speed=turn_speed,
            ff_time=ff_time,
        )

    def _update_yellow_decisions(self, idx: I, colors: npt.NDArray[np.int8]) -> None:
        f = self.fleet
        if self.signal.interval is Interval.GREEN:
            # Decisions only live through yellow/all-red; a vehicle already on the line keeps it.
            on_line = (f.stage[idx] == Stage.INBOUND) & (self._L - f.s[idx] < 0.5)
            f.yellow[idx] = np.where(on_line & (f.yellow[idx] >= YellowDecision.GO), f.yellow[idx], 0)
            return
        inb = idx[f.stage[idx] == Stage.INBOUND]
        undecided = inb[(colors[f.approach[inb]] == _YELLOW) & (f.yellow[inb] == YellowDecision.NONE)]
        if undecided.size == 0:
            return
        d = self._L - f.s[undecided]
        v = f.v[undecided]
        required_decel = v * v / (2.0 * np.maximum(d - STOP_LINE_SETBACK_M, 0.1))
        ttsl = d / np.maximum(v, 0.1)
        # Up to two queued permissive lefts clear on the change interval (HCM assumes ~2 per cycle).
        sneaker = (f.movement[undecided] == Movement.LEFT) & (d < SNEAKER_ZONE_M) & (v < 2.0)
        go = (required_decel > f.max_stop_decel[undecided]) | (ttsl <= f.yellow_ttsl[undecided])
        f.yellow[undecided] = np.where(
            sneaker, YellowDecision.SNEAK, np.where(go, YellowDecision.GO, YellowDecision.STOP)
        )

    def _leaders(self, idx: I) -> tuple[F, F, I]:
        """Gap and approach rate to each vehicle's leader, plus the ids of inbound lane heads.

        Vehicles are sorted by (link, position); a vehicle's leader is the next one on its link.
        Inbound lane heads follow the tail vehicle of their destination exit through the box.
        """
        f = self.fleet
        L = self._L
        inbound = f.stage[idx] == Stage.INBOUND
        s = f.s[idx]
        link = np.where(
            inbound, f.approach[idx].astype(np.intp) * INBOUND_LANES + f.lane[idx], _N_LINKS + f.out_link[idx]
        )
        coord = np.where(inbound, s, s - L - f.path_len[idx])

        order = np.lexsort((coord, link))
        sl, sc, si = link[order], coord[order], idx[order]
        same_next = np.append(sl[1:] == sl[:-1], False)
        first = np.insert(sl[1:] != sl[:-1], 0, True)

        gap_sorted = np.full(order.size, np.inf)
        dv_sorted = np.zeros(order.size)
        k = np.flatnonzero(same_next)
        gap_sorted[k] = sc[k + 1] - sc[k] - f.length[si[k + 1]]
        dv_sorted[k] = f.v[si[k]] - f.v[si[k + 1]]

        # Tail vehicle of every outbound link: an inbound lane head follows it through the box.
        tail_coord = np.full(_N_LINKS, np.inf)
        tail_len = np.zeros(_N_LINKS)
        tail_v = np.zeros(_N_LINKS)
        tails = np.flatnonzero(first & (sl >= _N_LINKS))
        tail_coord[sl[tails] - _N_LINKS] = sc[tails]
        tail_len[sl[tails] - _N_LINKS] = f.length[si[tails]]
        tail_v[sl[tails] - _N_LINKS] = f.v[si[tails]]

        heads = np.flatnonzero(~same_next & (sl < _N_LINKS))
        head_ids = si[heads]
        ol = f.out_link[head_ids].astype(np.intp)
        tc = tail_coord[ol]
        has_tail = np.isfinite(tc)
        h = heads[has_tail]
        hid = head_ids[has_tail]
        gap_sorted[h] = (L - f.s[hid]) + f.path_len[hid] + tc[has_tail] - tail_len[ol[has_tail]]
        dv_sorted[h] = f.v[hid] - tail_v[ol[has_tail]]

        gap = np.empty_like(gap_sorted)
        dv = np.empty_like(dv_sorted)
        gap[order] = gap_sorted
        dv[order] = dv_sorted
        return gap, dv, head_ids

    def _accelerations(self, idx: I, colors: npt.NDArray[np.int8]) -> F:
        f = self.fleet
        L = self._L
        stage = f.stage[idx]
        inbound = stage == Stage.INBOUND
        s = f.s[idx]
        v = f.v[idx]
        gap, dv, head_ids = self._leaders(idx)

        # Virtual stop-line obstacle.
        col = colors[f.approach[idx]]
        must_stop = inbound & (col != _GREEN) & (f.yellow[idx] < YellowDecision.GO)
        yield_ids = self._yielding_heads(head_ids, colors)
        if yield_ids.size:
            must_stop |= np.isin(idx, yield_ids)
        gap_stop = L - STOP_LINE_SETBACK_M + f.s0[idx] - s
        use_stop = must_stop & (gap_stop < gap)
        gap = np.where(use_stop, gap_stop, gap)
        dv = np.where(use_stop, v, dv)

        # Desired speed, anticipating turns.
        turning = f.movement[idx] != Movement.THROUGH
        ts = f.turn_speed[idx]
        v0 = f.v0[idx]
        dist = np.maximum(L - s, 0.0)
        v0 = np.where(inbound & turning, np.minimum(v0, np.sqrt(ts * ts + 2.0 * f.b[idx] * dist)), v0)
        v0 = np.where((stage == Stage.BOX) & turning, np.minimum(v0, ts), v0)

        acc = idm_acceleration(v, v0, gap, dv, f.T[idx], f.a[idx], f.b[idx], f.s0[idx])

        cruising = (v > 3.0) & (gap > 40.0)
        acc = acc + np.where(cruising, self._noise.normal(0.0, 0.12, idx.size), 0.0)
        return self._apply_reaction(idx, acc)

    def _apply_reaction(self, idx: I, acc: F) -> F:
        """Drivers at standstill wait their reaction time before moving off."""
        f = self.fleet
        timer = f.launch_timer[idx]
        stopped = f.v[idx] < 0.3
        wants = acc > 0.05
        start = stopped & wants & (timer < 0)
        timer = np.where(start, f.reaction[idx], timer)
        counting = stopped & wants & (timer > 0)
        acc = np.where(counting, 0.0, acc)
        timer = np.where(counting, np.maximum(timer - DT, 0.0), timer)
        timer = np.where(~stopped | ~wants, -1.0, timer)
        f.launch_timer[idx] = timer
        return acc

    def _yielding_heads(self, head_ids: I, colors: npt.NDArray[np.int8]) -> I:
        """Lane heads that must wait: permissive lefts without a gap, turners blocked by walkers."""
        f = self.fleet
        yielding: list[int] = []
        for i in head_ids:
            if self._L - f.s[i] > YIELD_LOOKAHEAD_M:
                continue
            approach = Approach.from_index(int(f.approach[i]))
            proceeding = colors[approach.ordinal] == _GREEN or f.yellow[i] >= YellowDecision.GO
            if not proceeding:
                continue
            movement = Movement(int(f.movement[i]))
            if movement is Movement.THROUGH:
                continue
            blocked = self.pedestrians.crosswalk_busy(exit_approach(approach, movement))
            if not blocked and movement is Movement.LEFT:
                blocked = self._opposing_conflict(approach, float(f.crit_gap[i]), colors)
            if blocked:
                yielding.append(int(i))
        return np.asarray(yielding, dtype=np.intp)

    def _opposing_conflict(
        self, approach: Approach, critical_gap: float, colors: npt.NDArray[np.int8]
    ) -> bool:
        f = self.fleet
        opp = approach.opposite
        idx = f.indices
        mine = idx[(f.approach[idx] == opp.ordinal) & (f.movement[idx] != Movement.LEFT)]
        if mine.size == 0:
            return False
        stage = f.stage[mine]
        in_box = mine[stage == Stage.BOX]
        if in_box.size and np.any(f.s[in_box] - self._L < 0.75 * f.path_len[in_box]):
            return True
        inbound = mine[(stage == Stage.INBOUND) & (f.lane[mine] == THROUGH_RIGHT_LANE)]
        if inbound.size == 0:
            return False
        opp_green = colors[opp.ordinal] == _GREEN
        will_cross = opp_green | (f.yellow[inbound] >= YellowDecision.GO)
        d = self._L - f.s[inbound]
        v = f.v[inbound]
        arriving = (v > 1.0) & (d / np.maximum(v, 0.1) < critical_gap)
        launching = opp_green & (d < 3.0)
        return bool(np.any(will_cross & (arriving | launching)))

    def _integrate(self, idx: I, acc: F) -> None:
        f = self.fleet
        f.acc[idx] = acc
        f.s[idx], f.v[idx] = ballistic_update(f.s[idx], f.v[idx], acc, DT)

    def _transitions(self, idx: I, colors: npt.NDArray[np.int8]) -> None:
        f = self.fleet
        L = self._L
        stage = f.stage[idx]
        s = f.s[idx]

        inbound = idx[stage == Stage.INBOUND]
        vin = f.v[inbound]
        f.wait_total[inbound] += np.where(vin < 0.5, DT, 0.0)
        new_stop = f.was_moving[inbound] & (vin < 0.5)
        f.n_stops[inbound] += new_stop.astype(np.int16)
        f.was_moving[inbound] = np.where(new_stop, False, f.was_moving[inbound] | (vin > 3.0))

        crossing = idx[(stage == Stage.INBOUND) & (s >= L)]
        for i in crossing:
            f.stage[i] = Stage.BOX
            f.crossed_time[i] = self.time
            a = int(f.approach[i])
            self.departures[a] += 1
            sneaker = f.yellow[i] == YellowDecision.SNEAK
            if colors[a] == _RED and f.cls[i] != VehicleClass.EMERGENCY and not sneaker:
                self.red_light_violations += 1

        exiting_box = idx[(stage == Stage.BOX) & (s >= L + f.path_len[idx])]
        for i in exiting_box:
            f.stage[i] = Stage.OUTBOUND
            delay = max(0.0, (self.time - f.arrival_time[i]) - f.ff_time[i])
            self.trips.append(
                TripRecord(
                    int(f.approach[i]), int(f.movement[i]), int(f.cls[i]), delay, int(f.n_stops[i]), self.time
                )
            )

        done = idx[(stage == Stage.OUTBOUND) & (s >= L + f.path_len[idx] + self._exit_len)]
        if done.size:
            f.remove(done)


def lane_id(approach: Approach, lane: int) -> int:
    return approach.ordinal * INBOUND_LANES + lane


__all__ = ["DT", "LEFT_LANE", "SimulationEngine", "TripRecord", "VehiclePoses", "lane_id", "resolve_timing"]
