"""Vehicle classes and heterogeneous driver behaviour.

Every driver gets individual Intelligent Driver Model parameters (Treiber et al., 2000), plus
behavioural traits used at the junction: reaction time on queue start-up, critical gap for
permissive left turns, and the yellow-onset stop/go threshold. Heterogeneity is what turns a
textbook model into realistic queue discharge. Start-up lost time and saturation headway
emerge from it instead of being scripted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jevflow.domain.enums import VehicleClass


@dataclass(frozen=True, slots=True)
class ClassSpec:
    length_m: float
    length_sd: float
    max_accel: float  # IDM a (m/s²)
    comfort_decel: float  # IDM b (m/s²)
    time_gap_s: float  # IDM T median
    speed_factor: float  # desired speed relative to the limit


CLASS_SPECS: dict[VehicleClass, ClassSpec] = {
    VehicleClass.CAR: ClassSpec(4.5, 0.3, 1.6, 2.2, 1.15, 1.03),
    VehicleClass.VAN: ClassSpec(5.6, 0.3, 1.3, 2.0, 1.3, 1.0),
    VehicleClass.BUS: ClassSpec(12.0, 0.3, 0.9, 1.6, 1.6, 0.9),
    VehicleClass.TRUCK: ClassSpec(10.5, 1.0, 0.8, 1.7, 1.7, 0.9),
    VehicleClass.EMERGENCY: ClassSpec(6.0, 0.2, 2.4, 3.0, 1.0, 1.25),
}


@dataclass(frozen=True, slots=True)
class Driver:
    vclass: VehicleClass
    length: float
    desired_speed: float  # m/s
    time_gap: float
    max_accel: float
    comfort_decel: float
    min_gap: float
    reaction_time: float  # delay before moving off from standstill
    critical_gap: float  # accepted time gap for permissive left turns
    yellow_ttsl: float  # time-to-stop-line above which the driver stops on yellow
    max_stop_decel: float  # hardest braking the driver accepts at yellow onset


def _lognormal(rng: np.random.Generator, median: float, sigma: float) -> float:
    return float(median * np.exp(rng.normal(0.0, sigma)))


def sample_vehicle_class(rng: np.random.Generator, heavy_share: float) -> VehicleClass:
    u = rng.random()
    if u < heavy_share:
        return VehicleClass.BUS if rng.random() < 0.35 else VehicleClass.TRUCK
    if u < heavy_share + 0.10:
        return VehicleClass.VAN
    return VehicleClass.CAR


def sample_driver(rng: np.random.Generator, vclass: VehicleClass, speed_limit_ms: float) -> Driver:
    spec = CLASS_SPECS[vclass]
    return Driver(
        vclass=vclass,
        length=float(np.clip(rng.normal(spec.length_m, spec.length_sd), 3.6, 16.5)),
        desired_speed=float(np.clip(rng.normal(spec.speed_factor, 0.07), 0.8, 1.35)) * speed_limit_ms,
        time_gap=float(np.clip(_lognormal(rng, spec.time_gap_s, 0.18), 0.75, 2.4)),
        max_accel=float(np.clip(_lognormal(rng, spec.max_accel, 0.15), 0.5, 3.0)),
        comfort_decel=float(np.clip(_lognormal(rng, spec.comfort_decel, 0.12), 1.2, 3.5)),
        min_gap=float(np.clip(rng.normal(1.8, 0.3), 1.0, 3.0)),
        reaction_time=float(np.clip(_lognormal(rng, 0.85, 0.3), 0.4, 2.2)),
        critical_gap=float(np.clip(rng.normal(4.6, 0.6), 3.2, 6.5)),
        yellow_ttsl=float(np.clip(rng.normal(3.1, 0.45), 2.0, 4.5)),
        max_stop_decel=float(np.clip(rng.normal(3.4, 0.5), 2.3, 5.0)),
    )
