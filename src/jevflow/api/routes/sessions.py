"""Live simulation sessions: create, control, inspect, and stream over WebSocket."""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from jevflow.api.deps import SessionDep, SessionsDep
from jevflow.api.schemas import CreateSessionIn, SessionControlIn, SessionOut
from jevflow.runtime.session import SimulationSession

router = APIRouter(tags=["sessions"])


def _out(session: SimulationSession) -> SessionOut:
    sample = session.latest_sample
    return SessionOut(
        **session.info(),
        kpi=sample.model_dump() if sample else None,
        summary=session.summary.model_dump() if session.summary else None,
    )


@router.post("/api/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionIn, manager: SessionsDep) -> SessionOut:
    session = manager.create(body.scenario, body.controller, body.seed, body.speed)
    return _out(session)


@router.get("/api/sessions")
async def list_sessions(manager: SessionsDep) -> list[SessionOut]:
    return [_out(s) for s in manager.list()]


@router.get("/api/sessions/{session_id}")
async def get_session(session: SessionDep) -> SessionOut:
    return _out(session)


@router.post("/api/sessions/{session_id}/control")
async def control_session(body: SessionControlIn, session: SessionDep) -> SessionOut:
    if body.speed is not None:
        session.set_speed(body.speed)
    match body.action:
        case "pause":
            session.pause()
        case "resume":
            session.resume()
        case "stop":
            session.stop()
        case None:
            pass
    return _out(session)


@router.get("/api/sessions/{session_id}/metrics")
async def session_metrics(session: SessionDep, since: float = Query(-1.0)) -> list[dict[str, Any]]:
    return [s.model_dump() for s in session.evaluator.samples if s.t > since]


@router.get("/api/sessions/{session_id}/decisions")
async def session_decisions(
    session: SessionDep, limit: int = Query(50, ge=1, le=300)
) -> list[dict[str, Any]]:
    return list(session.decision_log)[-limit:][::-1]


@router.websocket("/ws/sessions/{session_id}")
async def stream_session(websocket: WebSocket, session_id: str) -> None:
    manager = websocket.app.state.sessions
    session: SimulationSession | None = manager.get(session_id)
    if session is None:
        await websocket.close(code=4404, reason="session not found")
        return
    await websocket.accept()
    queue = session.subscribe()
    try:
        await websocket.send_json(session.init_message())
        await websocket.send_json(session.frame())
        if session.summary is not None:
            await websocket.send_json({"type": "summary", "summary": session.summary.model_dump()})
        while True:
            await websocket.send_json(await queue.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.unsubscribe(queue)
        with contextlib.suppress(RuntimeError):
            await websocket.close()
