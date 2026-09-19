"""Pre-generated arrival schedules.

Arrivals use Cowan's M3 headway model, the standard for urban approaches: a share of
vehicles travel bunched at the minimum headway Δ and the rest follow shifted-exponential
headways. The bunching share follows Akçelik & Chung (1994). An optional cosine modulation
reproduces platoons released by an upstream signal.

Generating the whole schedule up front, from dedicated random streams, guarantees that every
control strategy is evaluated against identical traffic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jevflow.domain.enums import Approach, Movement, VehicleClass
from jevflow.domain.geometry import LEFT_LANE, THROUGH_RIGHT_LANE
from jevflow.simulation.drivers import Driver, sample_driver, sample_vehicle_class
from jevflow.simulation.rng import RandomStreams
from jevflow.simulation.scenario import ApproachDemand, Scenario

MIN_HEADWAY_S = 1.5
BUNCHING_COEFF = 1.5  # Akçelik & Chung b for a single urban lane
MAX_DETECTOR_PASSES = 3


@dataclass(frozen=True, slots=True)
class Arrival:
    time: float
    approach: Approach
    lane: int
    movement: Movement
    driver: Driver
    missed_by: int  # bit mask of per-lane loops (stop, advance, entry) that will miss this vehicle


@dataclass(frozen=True, slots=True)
class PedestrianArrival:
    time: float
    leg: Approach  # the leg whose crosswalk is used
    side: int  # 0/1: which kerb the pedestrian starts from
    speed: float


def _cowan_headway(rng: np.random.Generator, q: float) -> float:
    """One M3 headway (s) at flow ``q`` veh/s."""
    q = min(q, 0.95 / MIN_HEADWAY_S)
    alpha = math.exp(-BUNCHING_COEFF * MIN_HEADWAY_S * q)  # proportion of free (unbunched) vehicles
    if rng.random() > alpha:
        return MIN_HEADWAY_S
    lam = alpha * q / (1.0 - MIN_HEADWAY_S * q)
    return MIN_HEADWAY_S + float(rng.exponential(1.0 / lam))


def _lane_arrival_times(
    rng: np.random.Generator, demand: ApproachDemand, share: float, duration: float, phase: float
) -> list[float]:
    times: list[float] = []
    t = float(rng.uniform(0, 4))
    while t < duration:
        q = demand.rate_vph(t) * share / 3600.0
        if demand.platoon_strength > 0:
            q *= 1 + demand.platoon_strength * math.cos(2 * math.pi * t / demand.upstream_cycle_s + phase)
        if q < 1e-4:
            t += 1.0
            continue
        times.append(t)
        t += _cowan_headway(rng, q)
    return times


def generate_arrivals(scenario: Scenario, streams: RandomStreams, speed_limit_ms: float) -> list[Arrival]:
    arrivals: list[Arrival] = []
    for approach, demand in scenario.demand.items():
        rng = streams.get(f"arrivals.{approach}")
        drivers = streams.get(f"drivers.{approach}")
        misses = streams.get(f"detector-miss.{approach}")
        phase = float(rng.uniform(0, 2 * math.pi))
        turning = demand.turning
        lanes = (
            (LEFT_LANE, turning.left),
            (THROUGH_RIGHT_LANE, turning.through + turning.right),
        )
        for lane, share in lanes:
            if share <= 0:
                continue
            for t in _lane_arrival_times(rng, demand, share, scenario.duration_s, phase):
                if lane == LEFT_LANE:
                    movement = Movement.LEFT
                else:
                    p_right = turning.right / (turning.through + turning.right)
                    movement = Movement.RIGHT if drivers.random() < p_right else Movement.THROUGH
                vclass = sample_vehicle_class(drivers, demand.heavy_vehicle_share)
                mask = sum(
                    1 << k for k in range(MAX_DETECTOR_PASSES) if misses.random() < scenario.sensors.miss_rate
                )
                arrivals.append(
                    Arrival(t, approach, lane, movement, sample_driver(drivers, vclass, speed_limit_ms), mask)
                )

    ev_rng = streams.get("emergency")
    for event in scenario.emergencies:
        driver = sample_driver(ev_rng, VehicleClass.EMERGENCY, speed_limit_ms)
        arrivals.append(Arrival(event.at_s, event.approach, THROUGH_RIGHT_LANE, Movement.THROUGH, driver, 0))

    arrivals.sort(key=lambda a: a.time)
    return arrivals


def generate_pedestrians(scenario: Scenario, streams: RandomStreams) -> list[PedestrianArrival]:
    peds: list[PedestrianArrival] = []
    for leg, per_hour in scenario.pedestrians_per_hour.items():
        if per_hour <= 0:
            continue
        rng = streams.get(f"pedestrians.{leg}")
        t = float(rng.exponential(3600.0 / per_hour))
        while t < scenario.duration_s:
            speed = float(np.clip(rng.normal(1.35, 0.22), 0.8, 1.9))
            peds.append(PedestrianArrival(t, leg, int(rng.integers(0, 2)), speed))
            t += float(rng.exponential(3600.0 / per_hour))
    peds.sort(key=lambda p: p.time)
    return peds
