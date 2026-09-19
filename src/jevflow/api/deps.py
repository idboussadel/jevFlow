"""FastAPI dependencies resolving application-scoped services from ``app.state``."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status

from jevflow.config import Settings
from jevflow.persistence.repository import Repository
from jevflow.runtime.manager import BenchmarkRunner, SessionManager
from jevflow.runtime.session import SimulationSession


def settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def repository(request: Request) -> Repository:
    return cast("Repository", request.app.state.repository)


def sessions(request: Request) -> SessionManager:
    return cast("SessionManager", request.app.state.sessions)


def benchmarks(request: Request) -> BenchmarkRunner:
    return cast("BenchmarkRunner", request.app.state.benchmarks)


def session_or_404(
    session_id: str, manager: Annotated[SessionManager, Depends(sessions)]
) -> SimulationSession:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"session {session_id!r} not found")
    return session


SettingsDep = Annotated[Settings, Depends(settings)]
RepositoryDep = Annotated[Repository, Depends(repository)]
SessionsDep = Annotated[SessionManager, Depends(sessions)]
BenchmarksDep = Annotated[BenchmarkRunner, Depends(benchmarks)]
SessionDep = Annotated[SimulationSession, Depends(session_or_404)]
