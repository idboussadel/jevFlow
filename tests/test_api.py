from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from jevflow.api.app import create_app
from jevflow.config import Settings


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(typesafe_api_key=None, database_path=tmp_path / "t.db")))


def test_health_and_catalogs(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/health").json()["jev_enabled"] is False
        assert len(client.get("/api/scenarios").json()) >= 5
        jev = next(c for c in client.get("/api/controllers").json() if c["key"] == "jev")
        assert jev["available"] is False
        assert client.post("/api/sessions", json={"controller": "jev"}).status_code == 400
        assert (
            client.post("/api/sessions", json={"scenario": "nope", "controller": "actuated"}).status_code
            == 404
        )


def test_live_session_streams_and_persists(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        session = client.post("/api/sessions", json={"controller": "actuated", "speed": 20}).json()
        with client.websocket_connect(f"/ws/sessions/{session['id']}") as ws:
            assert ws.receive_json()["type"] == "init"
            kinds = {ws.receive_json()["type"] for _ in range(30)}
        assert "frame" in kinds
        client.post(f"/api/sessions/{session['id']}/control", json={"action": "stop"})
        for _ in range(50):
            if client.get(f"/api/sessions/{session['id']}").json()["status"] == "stopped":
                break
            time.sleep(0.05)
        run = client.get(f"/api/runs/{session['id']}").json()
        assert run["status"] == "stopped" and run["summary"]["decisions"] > 0
        assert client.get(f"/api/runs/{session['id']}/metrics").json()
