"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevflow.simulation.scenario import Scenario


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    jev_enabled: bool
    jev_model: str


class ScenarioOut(BaseModel):
    key: str
    name: str
    tagline: str
    description: str
    duration_s: float
    approach_length_m: float
    peak_demand_vph: dict[str, float]
    pedestrians_per_hour: dict[str, float]
    emergencies: int
    detector_faults: int

    @classmethod
    def of(cls, s: Scenario) -> ScenarioOut:
        return cls(
            key=s.key,
            name=s.name,
            tagline=s.tagline,
            description=s.description,
            duration_s=s.duration_s,
            approach_length_m=s.approach_length_m,
            peak_demand_vph={a.value: d.peak_vph for a, d in s.demand.items()},
            pedestrians_per_hour={a.value: r for a, r in s.pedestrians_per_hour.items()},
            emergencies=len(s.emergencies),
            detector_faults=len(s.sensors.faults),
        )


class ControllerOut(BaseModel):
    key: str
    name: str
    description: str
    available: bool
    reason: str | None = None


class CreateSessionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = "rush_hour"
    controller: str = "jev"
    seed: int = Field(42, ge=0, le=2**31 - 1)
    speed: float = Field(1.0, ge=0.1, le=50.0)


class SessionControlIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["pause", "resume", "stop"] | None = None
    speed: float | None = Field(None, ge=0.1, le=50.0)


class SessionOut(BaseModel):
    id: str
    scenario: str
    scenario_name: str
    controller: str
    controller_name: str
    seed: int
    status: str
    speed: float
    time: float
    duration: float
    error: str | None = None
    kpi: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None


class BenchmarkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = "rush_hour"
    controllers: list[str] = Field(
        default_factory=lambda: ["fixed_time", "actuated", "max_pressure", "jev"], min_length=1
    )
    seed: int = Field(42, ge=0, le=2**31 - 1)
    duration_s: float | None = Field(None, ge=60, le=3600)
