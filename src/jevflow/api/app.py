"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from jevflow import __version__
from jevflow.api.routes import history, meta, sessions
from jevflow.config import Settings, get_settings
from jevflow.control.factory import StrategyUnavailableError
from jevflow.persistence.repository import Repository
from jevflow.runtime.manager import BenchmarkRunner, CapacityError, SessionManager


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        repository = Repository(settings.database_path)
        app.state.settings = settings
        app.state.repository = repository
        app.state.sessions = SessionManager(settings, repository)
        app.state.benchmarks = BenchmarkRunner(settings, repository)
        logging.getLogger(__name__).info(
            "JevFlow API ready (Jev %s)", "on" if settings.jev_enabled else "off"
        )
        try:
            yield
        finally:
            await app.state.sessions.shutdown()
            repository.close()

    app = FastAPI(
        title="JevFlow API",
        version=__version__,
        summary="Real-time traffic-signal control with TypeSafe Jev",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(meta.router)
    app.include_router(sessions.router)
    app.include_router(history.router)

    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc.args[0]) if exc.args else "not found"}, status.HTTP_404_NOT_FOUND
        )

    @app.exception_handler(CapacityError)
    async def _capacity(_: Request, exc: CapacityError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status.HTTP_429_TOO_MANY_REQUESTS)

    @app.exception_handler(StrategyUnavailableError)
    async def _unavailable(_: Request, exc: StrategyUnavailableError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status.HTTP_400_BAD_REQUEST)

    return app


app = create_app()
