"""Lifecycle management for live sessions and benchmark jobs."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from jevflow.config import Settings
from jevflow.control.factory import create_strategy
from jevflow.persistence.repository import Repository
from jevflow.runtime.evaluation import RunSummary
from jevflow.runtime.session import SimulationSession
from jevflow.simulation.engine import resolve_timing
from jevflow.simulation.scenario import Scenario
from jevflow.simulation.scenarios import get_scenario

log = logging.getLogger(__name__)


class CapacityError(RuntimeError):
    pass


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def build_session(
    scenario: Scenario,
    controller: str,
    seed: int,
    settings: Settings,
    *,
    realtime: bool,
    speed: float = 1.0,
    repository: Repository | None = None,
    mode: str = "live",
    benchmark_id: str | None = None,
) -> SimulationSession:
    strategy = create_strategy(controller, scenario, resolve_timing(scenario), settings)
    return SimulationSession(
        _new_id("run"),
        scenario,
        strategy,
        seed,
        realtime=realtime,
        speed=speed,
        fps=settings.stream_fps,
        repository=repository,
        mode=mode,
        benchmark_id=benchmark_id,
    )


class SessionManager:
    def __init__(self, settings: Settings, repository: Repository) -> None:
        self._settings = settings
        self._repo = repository
        self._sessions: dict[str, SimulationSession] = {}
        self._tasks: dict[str, asyncio.Task[RunSummary | None]] = {}

    def create(self, scenario_key: str, controller: str, seed: int, speed: float) -> SimulationSession:
        running = [s for s in self._sessions.values() if not s.status.terminal]
        if len(running) >= self._settings.max_live_sessions:
            raise CapacityError(f"at most {self._settings.max_live_sessions} live sessions may run at once")
        self._evict_finished()
        session = build_session(
            get_scenario(scenario_key), controller, seed, self._settings,
            realtime=True, speed=speed, repository=self._repo,
        )  # fmt: skip
        self._sessions[session.id] = session
        self._tasks[session.id] = asyncio.create_task(session.run(), name=f"session:{session.id}")
        return session

    def get(self, session_id: str) -> SimulationSession | None:
        return self._sessions.get(session_id)

    def list(self) -> list[SimulationSession]:
        return list(self._sessions.values())

    def _evict_finished(self, keep: int = 8) -> None:
        finished = [s for s in self._sessions.values() if s.status.terminal]
        for session in finished[:-keep] if len(finished) > keep else []:
            self._sessions.pop(session.id, None)
            self._tasks.pop(session.id, None)

    async def shutdown(self) -> None:
        for session in self._sessions.values():
            session.stop()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)


@dataclass(slots=True)
class BenchmarkJob:
    id: str
    scenario: str
    seed: int
    controllers: list[str]
    duration_s: float | None
    status: str = "running"
    progress: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario": self.scenario,
            "seed": self.seed,
            "controllers": self.controllers,
            "duration_s": self.duration_s,
            "status": self.status,
            "progress": self.progress,
            "results": self.results,
            "error": self.error,
        }


def run_headless(
    scenario: Scenario,
    controller: str,
    seed: int,
    settings: Settings,
    repository: Repository | None = None,
    benchmark_id: str | None = None,
) -> tuple[str, RunSummary | None, str | None]:
    """Run one headless session to completion in its own event loop (thread-friendly)."""
    session = build_session(
        scenario, controller, seed, settings,
        realtime=False, repository=repository, mode="benchmark", benchmark_id=benchmark_id,
    )  # fmt: skip
    summary = asyncio.run(session.run())
    return session.id, summary, session.error


class BenchmarkRunner:
    """Runs every controller on the *same* scenario and seed: common random numbers."""

    def __init__(self, settings: Settings, repository: Repository) -> None:
        self._settings = settings
        self._repo = repository
        self._jobs: dict[str, BenchmarkJob] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def start(
        self, scenario_key: str, controllers: list[str], seed: int, duration_s: float | None
    ) -> BenchmarkJob:
        scenario = get_scenario(scenario_key)
        job = BenchmarkJob(_new_id("bench"), scenario_key, seed, controllers, duration_s)
        job.progress = dict.fromkeys(controllers, "queued")
        self._jobs[job.id] = job
        self._repo.create_benchmark(
            job.id, scenario_key, seed, {"controllers": controllers, "duration_s": duration_s}
        )
        if duration_s is not None:
            scenario = scenario.model_copy(update={"duration_s": duration_s})
        task = asyncio.create_task(self._run(job, scenario), name=f"benchmark:{job.id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def get(self, job_id: str) -> BenchmarkJob | None:
        return self._jobs.get(job_id)

    async def _run(self, job: BenchmarkJob, scenario: Scenario) -> None:
        try:
            for controller in job.controllers:
                job.progress[controller] = "running"
                run_id, summary, error = await asyncio.to_thread(
                    run_headless, scenario, controller, job.seed, self._settings, self._repo, job.id
                )
                job.progress[controller] = "failed" if error else "done"
                job.results[controller] = {
                    "run_id": run_id,
                    "summary": summary.model_dump() if summary else None,
                    "error": error,
                }
            job.status = "finished"
        except Exception as exc:
            log.exception("benchmark %s failed", job.id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        await asyncio.to_thread(self._repo.finish_benchmark, job.id, job.status, job.results)
