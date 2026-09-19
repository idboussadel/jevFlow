"""SQLite persistence for runs, one-second metric samples, decisions and benchmarks.

A single connection in WAL mode, guarded by a lock, is plenty for this workload and keeps the
project dependency-free. Structured payloads are stored as JSON so the schema stays stable while
metrics evolve. Async callers go through :func:`asyncio.to_thread` so the event loop never blocks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    scenario      TEXT NOT NULL,
    controller    TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    mode          TEXT NOT NULL,
    status        TEXT NOT NULL,
    benchmark_id  TEXT,
    summary       TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS metric_samples (
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    t       REAL NOT NULL,
    data    TEXT NOT NULL,
    PRIMARY KEY (run_id, t)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS decisions (
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    t           REAL NOT NULL,
    action      TEXT NOT NULL,
    applied     TEXT NOT NULL,
    source      TEXT NOT NULL,
    confidence  REAL,
    latency_ms  REAL,
    data        TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS benchmarks (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    scenario    TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    status      TEXT NOT NULL,
    config      TEXT NOT NULL,
    results     TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Repository:
    def __init__(self, path: Path) -> None:
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        # Pragmas must run outside a transaction; executescript manages its own.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                yield cur
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            finally:
                cur.close()

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        run_id: str,
        scenario: str,
        controller: str,
        seed: int,
        mode: str,
        benchmark_id: str | None = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO runs (id, created_at, scenario, controller, seed, mode, status, benchmark_id)"
                " VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
                (run_id, _now(), scenario, controller, seed, mode, benchmark_id),
            )

    def finish_run(self, run_id: str, status: str, summary: dict[str, Any] | None) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE runs SET status = ?, summary = ? WHERE id = ?",
                (status, json.dumps(summary) if summary is not None else None, run_id),
            )

    def add_samples(self, run_id: str, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        with self._tx() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO metric_samples (run_id, t, data) VALUES (?, ?, ?)",
                [(run_id, s["t"], json.dumps(s)) for s in samples],
            )

    def add_decisions(self, run_id: str, decisions: list[dict[str, Any]]) -> None:
        if not decisions:
            return
        with self._tx() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO decisions"
                " (run_id, seq, t, action, applied, source, confidence, latency_ms, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        d["seq"],
                        d["t"],
                        d["action"],
                        d["applied"],
                        d["source"],
                        d.get("confidence"),
                        d.get("latency_ms"),
                        json.dumps(d),
                    )
                    for d in decisions
                ],
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [_run(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM runs WHERE id = ?", (run_id,))
        return _run(rows[0]) if rows else None

    def run_samples(self, run_id: str, since: float = -1.0) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT data FROM metric_samples WHERE run_id = ? AND t > ? ORDER BY t", (run_id, since)
        )
        return [json.loads(r["data"]) for r in rows]

    def run_decisions(self, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT data FROM decisions WHERE run_id = ? ORDER BY seq DESC LIMIT ?", (run_id, limit)
        )
        return [json.loads(r["data"]) for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    # ------------------------------------------------------------------ benchmarks
    def create_benchmark(self, benchmark_id: str, scenario: str, seed: int, config: dict[str, Any]) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO benchmarks (id, created_at, scenario, seed, status, config)"
                " VALUES (?, ?, ?, ?, 'running', ?)",
                (benchmark_id, _now(), scenario, seed, json.dumps(config)),
            )

    def finish_benchmark(self, benchmark_id: str, status: str, results: dict[str, Any] | None) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE benchmarks SET status = ?, results = ? WHERE id = ?",
                (status, json.dumps(results) if results is not None else None, benchmark_id),
            )

    def list_benchmarks(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT ?", (limit,))
        return [_benchmark(row) for row in rows]

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,))
        return _benchmark(rows[0]) if rows else None


def _run(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["summary"] = json.loads(out["summary"]) if out["summary"] else None
    return out


def _benchmark(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["config"] = json.loads(out["config"])
    out["results"] = json.loads(out["results"]) if out["results"] else None
    return out
