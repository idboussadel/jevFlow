"""Health, scenarios and controllers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from jevflow import __version__
from jevflow.api.deps import SettingsDep
from jevflow.api.schemas import ControllerOut, HealthOut, ScenarioOut
from jevflow.control.factory import strategy_catalog
from jevflow.simulation.scenarios import SCENARIOS

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
async def health(settings: SettingsDep) -> HealthOut:
    return HealthOut(version=__version__, jev_enabled=settings.jev_enabled, jev_model=settings.jev_model)


@router.get("/scenarios")
async def list_scenarios() -> list[ScenarioOut]:
    return [ScenarioOut.of(s) for s in SCENARIOS.values()]


@router.get("/scenarios/{key}")
async def get_scenario(key: str) -> ScenarioOut:
    scenario = SCENARIOS.get(key)
    if scenario is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"scenario {key!r} not found")
    return ScenarioOut.of(scenario)


@router.get("/controllers")
async def list_controllers(settings: SettingsDep) -> list[ControllerOut]:
    return [ControllerOut(**asdict(info)) for info in strategy_catalog(settings)]
