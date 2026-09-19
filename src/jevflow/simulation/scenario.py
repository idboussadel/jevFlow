"""Declarative scenario definitions: demand, pedestrians, incidents and sensor quality."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevflow.domain.enums import Approach
from jevflow.domain.timing import SignalTiming


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DemandPoint(_Model):
    at_s: float = Field(ge=0)
    vph: float = Field(ge=0, description="Approach flow rate in vehicles per hour")


class TurningShares(_Model):
    left: float = Field(ge=0, le=1)
    through: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> TurningShares:
        if abs(self.left + self.through + self.right - 1.0) > 1e-6:
            raise ValueError("turning shares must sum to 1")
        return self


class ApproachDemand(_Model):
    profile: tuple[DemandPoint, ...] = Field(min_length=1)
    turning: TurningShares = TurningShares(left=0.15, through=0.72, right=0.13)
    heavy_vehicle_share: float = Field(0.04, ge=0, le=0.5)
    platoon_strength: float = Field(0.0, ge=0, le=0.9, description="Arrival pulsing from an upstream signal")
    upstream_cycle_s: float = Field(90.0, gt=0)

    def rate_vph(self, t: float) -> float:
        xs = [p.at_s for p in self.profile]
        ys = [p.vph for p in self.profile]
        return float(np.interp(t, xs, ys))

    @property
    def peak_vph(self) -> float:
        return max(p.vph for p in self.profile)


DetectorKind = Literal["stop", "advance", "entry"]


class FaultEvent(_Model):
    """A scheduled detector failure, e.g. a loop stuck on after a cable fault."""

    approach: Approach
    lane: int = Field(ge=0, le=1)
    detector: DetectorKind
    mode: Literal["stuck_on", "dead", "chatter"]
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)


class SensorProfile(_Model):
    miss_rate: float = Field(0.012, ge=0, le=0.5, description="Probability a loop misses a vehicle")
    false_actuations_per_hour: float = Field(3.0, ge=0)
    faults: tuple[FaultEvent, ...] = ()


class EmergencyEvent(_Model):
    at_s: float = Field(ge=0)
    approach: Approach


class Scenario(_Model):
    key: str
    name: str
    tagline: str
    description: str
    duration_s: float = Field(900.0, gt=0)
    demand: dict[Approach, ApproachDemand]
    pedestrians_per_hour: dict[Approach, float] = Field(default_factory=dict)
    emergencies: tuple[EmergencyEvent, ...] = ()
    sensors: SensorProfile = SensorProfile()
    timing: SignalTiming | None = None
    approach_length_m: float = Field(250.0, ge=150, le=600)

    @model_validator(mode="after")
    def _all_approaches(self) -> Scenario:
        missing = set(Approach) - set(self.demand)
        if missing:
            raise ValueError(f"demand missing for {sorted(missing)}")
        return self
