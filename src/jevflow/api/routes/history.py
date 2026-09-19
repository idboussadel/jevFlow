"""Benchmarks and persisted run history."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from jevflow.api.deps import BenchmarksDep, RepositoryDep
from jevflow.api.schemas import BenchmarkIn
from jevflow.simulation.scenarios import get_scenario

router = APIRouter(prefix="/api", tags=["history"])


@router.post("/benchmarks", status_code=status.HTTP_202_ACCEPTED)
async def start_benchmark(body: BenchmarkIn, runner: BenchmarksDep) -> dict[str, Any]:
    get_scenario(body.scenario)  # validate early
    return runner.start(body.scenario, body.controllers, body.seed, body.duration_s).view()


@router.get("/benchmarks")
async def list_benchmarks(repo: RepositoryDep, limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
    return await asyncio.to_thread(repo.list_benchmarks, limit)


@router.get("/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str, runner: BenchmarksDep, repo: RepositoryDep) -> dict[str, Any]:
    job = runner.get(benchmark_id)
    if job is not None:
        return job.view()
    stored = await asyncio.to_thread(repo.get_benchmark, benchmark_id)
    if stored is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"benchmark {benchmark_id!r} not found")
    return {**stored, "progress": {}, "controllers": stored["config"]["controllers"]}


@router.get("/runs")
async def list_runs(repo: RepositoryDep, limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    return await asyncio.to_thread(repo.list_runs, limit)


@router.get("/runs/{run_id}")
async def get_run(run_id: str, repo: RepositoryDep) -> dict[str, Any]:
    run = await asyncio.to_thread(repo.get_run, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id!r} not found")
    return run


@router.get("/runs/{run_id}/metrics")
async def run_metrics(run_id: str, repo: RepositoryDep) -> list[dict[str, Any]]:
    return await asyncio.to_thread(repo.run_samples, run_id)


@router.get("/runs/{run_id}/decisions")
async def run_decisions(
    run_id: str, repo: RepositoryDep, limit: int = Query(200, ge=1, le=2000)
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(repo.run_decisions, run_id, limit)


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(run_id: str, repo: RepositoryDep) -> None:
    await asyncio.to_thread(repo.delete_run, run_id)
