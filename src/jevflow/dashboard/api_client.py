"""Thin, typed HTTP client the dashboard uses to talk to the JevFlow API."""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class JevFlowApi:
    def __init__(self, base_url: str, timeout: float = 8.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Cannot reach the JevFlow API at {self._client.base_url} ({exc.__class__.__name__})."
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(str(detail), response.status_code)
        return response.json() if response.content else None

    # meta
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")  # type: ignore[no-any-return]

    def scenarios(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/scenarios")  # type: ignore[no-any-return]

    def controllers(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/controllers")  # type: ignore[no-any-return]

    # sessions
    def create_session(self, scenario: str, controller: str, seed: int, speed: float) -> dict[str, Any]:
        body = {"scenario": scenario, "controller": controller, "seed": seed, "speed": speed}
        return self._request("POST", "/api/sessions", json=body)  # type: ignore[no-any-return]

    def session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/sessions/{session_id}")  # type: ignore[no-any-return]

    def control(
        self, session_id: str, action: str | None = None, speed: float | None = None
    ) -> dict[str, Any]:
        body = {k: v for k, v in {"action": action, "speed": speed}.items() if v is not None}
        return self._request("POST", f"/api/sessions/{session_id}/control", json=body)  # type: ignore[no-any-return]

    def session_metrics(self, session_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/sessions/{session_id}/metrics")  # type: ignore[no-any-return]

    def session_decisions(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/sessions/{session_id}/decisions", params={"limit": limit})  # type: ignore[no-any-return]

    # benchmarks & history
    def start_benchmark(
        self, scenario: str, controllers: list[str], seed: int, duration_s: float | None
    ) -> dict[str, Any]:
        body = {"scenario": scenario, "controllers": controllers, "seed": seed, "duration_s": duration_s}
        return self._request("POST", "/api/benchmarks", json=body)  # type: ignore[no-any-return]

    def benchmark(self, benchmark_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/benchmarks/{benchmark_id}")  # type: ignore[no-any-return]

    def benchmarks(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/benchmarks")  # type: ignore[no-any-return]

    def runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", "/api/runs", params={"limit": limit})  # type: ignore[no-any-return]

    def run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/runs/{run_id}/metrics")  # type: ignore[no-any-return]

    def run_decisions(self, run_id: str, limit: int = 300) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/runs/{run_id}/decisions", params={"limit": limit})  # type: ignore[no-any-return]
