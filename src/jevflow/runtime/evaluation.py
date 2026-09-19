"""Ground-truth evaluation. The only place allowed to look at truth, because it grades controllers.

Metrics follow the Highway Capacity Manual where one exists: control delay per vehicle,
signalised-intersection level of service (LOS) thresholds, and stops per vehicle. Vehicles
still inside the network when the run ends are counted with the delay accrued so far, so a
controller cannot look good by starving one approach until the clock runs out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict

from jevflow.control.base import Decision
from jevflow.domain.enums import Approach, VehicleClass
from jevflow.sensing.aggregator import TrafficStateAggregator
from jevflow.simulation.engine import SimulationEngine

LOS_THRESHOLDS = ((10, "A"), (20, "B"), (35, "C"), (55, "D"), (80, "E"))


def level_of_service(delay_s: float) -> str:
    for limit, grade in LOS_THRESHOLDS:
        if delay_s <= limit:
            return grade
    return "F"


def jain_fairness(values: list[float]) -> float:
    """Jain's index over per-approach delays: 1.0 = perfectly even, 1/n = one approach bears all."""
    x = np.asarray([v for v in values if v > 0], dtype=float)
    if x.size == 0:
        return 1.0
    return float(x.sum() ** 2 / (x.size * (x**2).sum()))


class MetricSample(BaseModel):
    """One-second snapshot for live charts and history."""

    model_config = ConfigDict(frozen=True)

    t: float
    phase: str
    interval: str
    true_queue: dict[str, int]
    estimated_queue: dict[str, float]
    off_map_queue: dict[str, int]
    vehicles_in_network: int
    avg_delay_recent_s: float
    avg_delay_cumulative_s: float
    throughput_vph: float
    max_wait_s: float
    served: int


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    simulated_seconds: float
    vehicles_served: int
    vehicles_unfinished: int
    avg_control_delay_s: float
    p50_delay_s: float
    p95_delay_s: float
    level_of_service: str
    stops_per_vehicle: float
    throughput_vph: float
    avg_queue_vehicles: float
    max_queue_vehicles: dict[str, int]
    approach_delay_s: dict[str, float]
    fairness_index: float
    max_wait_s: float
    spillback_seconds: float
    red_light_violations: int
    pedestrians_served: int
    pedestrian_avg_wait_s: float
    pedestrian_max_wait_s: float
    emergency_avg_delay_s: float | None
    phase_switches: int
    decisions: int
    decisions_by_source: dict[str, int]
    jev_calls: int
    jev_avg_latency_ms: float | None
    jev_p95_latency_ms: float | None
    jev_avg_confidence: float | None
    jev_input_tokens: int
    queue_estimate_mae: float


@dataclass(slots=True)
class _Accumulators:
    queue_sum: float = 0.0
    samples: int = 0
    spillback_s: float = 0.0
    est_abs_err: float = 0.0
    max_wait: float = 0.0


class Evaluator:
    def __init__(self, engine: SimulationEngine, aggregator: TrafficStateAggregator) -> None:
        self._engine = engine
        self._aggregator = aggregator
        self._acc = _Accumulators()
        self._max_queue = dict.fromkeys(Approach, 0)
        self.samples: deque[MetricSample] = deque(maxlen=7200)

    def sample(self) -> MetricSample:
        engine = self._engine
        t = engine.time
        truth = engine.true_queues()
        estimate = self._aggregator.estimated_queues()
        vertical = engine.vertical_queue()
        waits = engine.current_waits()
        trips = engine.trips
        recent = [tr.delay for tr in trips if t - tr.completed_at <= 60.0]
        window = [tr for tr in trips if t - tr.completed_at <= 300.0]
        acc = self._acc
        acc.samples += 1
        acc.queue_sum += sum(truth.values())
        acc.spillback_s += sum(1.0 for a in Approach if vertical[a] > 0)
        acc.est_abs_err += sum(abs(estimate[a] - truth[a]) for a in Approach) / len(Approach)
        acc.max_wait = max(acc.max_wait, *waits.values())
        for a in Approach:
            self._max_queue[a] = max(self._max_queue[a], truth[a])
        sample = MetricSample(
            t=round(t, 1),
            phase=engine.signal.phase.value,
            interval=engine.signal.interval.value,
            true_queue={a.value: truth[a] for a in Approach},
            estimated_queue={a.value: round(estimate[a], 1) for a in Approach},
            off_map_queue={a.value: vertical[a] for a in Approach},
            vehicles_in_network=len(engine.fleet) + sum(vertical.values()),
            avg_delay_recent_s=round(float(np.mean(recent)), 1) if recent else 0.0,
            avg_delay_cumulative_s=round(float(np.mean([tr.delay for tr in trips])), 1) if trips else 0.0,
            throughput_vph=round(len(window) * 3600.0 / min(max(t, 1.0), 300.0), 0),
            max_wait_s=round(max(waits.values()), 1),
            served=len(trips),
        )
        self.samples.append(sample)
        return sample

    def summary(self, decisions: list[Decision]) -> RunSummary:
        engine = self._engine
        t = max(engine.time, 1.0)
        trips = engine.trips
        delays = [tr.delay for tr in trips] + engine.unfinished_delay()
        d = np.asarray(delays) if delays else np.zeros(1)
        per_approach = {
            a.value: round(float(np.mean([tr.delay for tr in trips if tr.approach == a.ordinal] or [0.0])), 1)
            for a in Approach
        }
        ev = [tr.delay for tr in trips if tr.vclass == VehicleClass.EMERGENCY]
        peds = engine.pedestrians.stats
        by_source: dict[str, int] = {}
        for dec in decisions:
            by_source[dec.source.value] = by_source.get(dec.source.value, 0) + 1
        jev_calls = [dec for dec in decisions if dec.jev is not None]
        latencies = [dec.latency_ms for dec in jev_calls if dec.latency_ms is not None]
        confidences = [dec.confidence for dec in jev_calls if dec.confidence is not None]
        tokens = sum(dec.jev.input_tokens or 0 for dec in jev_calls if dec.jev)
        avg_delay = float(d.mean())
        return RunSummary(
            simulated_seconds=round(t, 1),
            vehicles_served=len(trips),
            vehicles_unfinished=len(delays) - len(trips),
            avg_control_delay_s=round(avg_delay, 1),
            p50_delay_s=round(float(np.percentile(d, 50)), 1),
            p95_delay_s=round(float(np.percentile(d, 95)), 1),
            level_of_service=level_of_service(avg_delay),
            stops_per_vehicle=round(float(np.mean([tr.stops for tr in trips])) if trips else 0.0, 2),
            throughput_vph=round(len(trips) * 3600.0 / t, 0),
            avg_queue_vehicles=round(self._acc.queue_sum / max(self._acc.samples, 1), 1),
            max_queue_vehicles={a.value: q for a, q in self._max_queue.items()},
            approach_delay_s=per_approach,
            fairness_index=round(jain_fairness(list(per_approach.values())), 3),
            max_wait_s=round(self._acc.max_wait, 1),
            spillback_seconds=round(self._acc.spillback_s, 0),
            red_light_violations=engine.red_light_violations,
            pedestrians_served=peds.served,
            pedestrian_avg_wait_s=round(peds.total_wait_s / peds.served, 1) if peds.served else 0.0,
            pedestrian_max_wait_s=round(peds.max_wait_s, 1),
            emergency_avg_delay_s=round(float(np.mean(ev)), 1) if ev else None,
            phase_switches=len(engine.signal.phase_changes),
            decisions=len(decisions),
            decisions_by_source=by_source,
            jev_calls=len(jev_calls),
            jev_avg_latency_ms=round(float(np.mean(latencies)), 1) if latencies else None,
            jev_p95_latency_ms=round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            jev_avg_confidence=round(float(np.mean(confidences)), 3) if confidences else None,
            jev_input_tokens=tokens,
            queue_estimate_mae=round(self._acc.est_abs_err / max(self._acc.samples, 1), 2),
        )
