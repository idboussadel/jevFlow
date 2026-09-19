"""A simulation session: truth → detectors → aggregator → strategy → signal controller.

Two execution modes share the same pipeline:

* **Live** (``realtime=True``): simulated time is paced against the wall clock at a chosen speed
  and frames are streamed to subscribers. Strategy calls run *concurrently* with the
  simulation, so Jev's network latency is real: the intersection keeps moving while the answer
  is in flight, and a stale answer (the phase changed meanwhile) is rejected.
* **Headless** (``realtime=False``): as fast as possible for benchmarks. Each decision is awaited
  inline, so results are reproducible for a given seed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from enum import StrEnum
from typing import Any

import numpy as np

from jevflow.control.base import ControlStrategy, Decision, DecisionContext
from jevflow.control.jev.state import action_label
from jevflow.control.safety import legal_actions, needs_decision, vet
from jevflow.domain.enums import Approach, ControlAction, DecisionSource, Interval
from jevflow.domain.geometry import INBOUND_LANES
from jevflow.persistence.repository import Repository
from jevflow.runtime.evaluation import Evaluator, MetricSample, RunSummary
from jevflow.sensing.aggregator import TrafficStateAggregator
from jevflow.sensing.detectors import DetectorArray
from jevflow.simulation.engine import SimulationEngine
from jevflow.simulation.scenario import Scenario

log = logging.getLogger(__name__)

MAX_STEPS_PER_ITERATION = 800
FLUSH_EVERY_S = 5.0
SUBSCRIBER_QUEUE = 8


class SessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (SessionStatus.FINISHED, SessionStatus.STOPPED, SessionStatus.FAILED)


class SimulationSession:
    def __init__(
        self,
        session_id: str,
        scenario: Scenario,
        strategy: ControlStrategy,
        seed: int,
        *,
        realtime: bool = True,
        speed: float = 1.0,
        fps: float = 15.0,
        repository: Repository | None = None,
        mode: str = "live",
        benchmark_id: str | None = None,
    ) -> None:
        self.id = session_id
        self.scenario = scenario
        self.strategy = strategy
        self.seed = seed
        self.realtime = realtime
        self.mode = mode
        self.benchmark_id = benchmark_id
        self.engine = SimulationEngine(scenario, seed)
        self.detectors = DetectorArray(self.engine, scenario.sensors)
        self.aggregator = TrafficStateAggregator(self.engine, self.detectors)
        self.evaluator = Evaluator(self.engine, self.aggregator)
        self.status = SessionStatus.CREATED
        self.error: str | None = None
        self.summary: RunSummary | None = None
        self.decisions: list[Decision] = []
        self.decision_log: deque[dict[str, Any]] = deque(maxlen=300)
        self.latest_sample: MetricSample | None = None
        self._repo = repository
        self._speed = speed
        self._fps = fps
        self._resume = asyncio.Event()
        self._resume.set()
        self._stop = False
        self._pending: asyncio.Task[Decision] | None = None
        self._pending_meta: tuple[DecisionContext, int] | None = None
        self._next_epoch = 0.0
        self._last_second = -1
        self._samples_buffer: list[dict[str, Any]] = []
        self._decisions_buffer: list[dict[str, Any]] = []
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._detector_ids = list(self.detectors.loops)

    # ================================================================== control
    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, min(speed, 50.0))

    def pause(self) -> None:
        if self.status is SessionStatus.RUNNING:
            self._resume.clear()
            self.status = SessionStatus.PAUSED
            self._broadcast({"type": "status", **self.info()})

    def resume(self) -> None:
        if self.status is SessionStatus.PAUSED:
            self.status = SessionStatus.RUNNING
            self._resume.set()
            self._broadcast({"type": "status", **self.info()})

    def stop(self) -> None:
        self._stop = True
        self._resume.set()

    # ================================================================== lifecycle
    async def run(self) -> RunSummary | None:
        self.status = SessionStatus.RUNNING
        await self._persist(lambda r: r.create_run(
            self.id, self.scenario.key, self.strategy.key, self.seed, self.mode, self.benchmark_id
        ))  # fmt: skip
        try:
            if self.realtime:
                await self._run_realtime()
            else:
                await self._run_headless()
            self.status = SessionStatus.FINISHED if self.engine.finished else SessionStatus.STOPPED
        except asyncio.CancelledError:
            self.status = SessionStatus.STOPPED
            raise
        except Exception as exc:
            log.exception("session %s failed", self.id)
            self.status = SessionStatus.FAILED
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            await self._finalise()
        return self.summary

    async def _finalise(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            with contextlib.suppress(BaseException):
                await self._pending
        self.summary = self.evaluator.summary(self.decisions)
        await self._flush()
        summary = self.summary.model_dump()
        await self._persist(lambda r: r.finish_run(self.id, self.status.value, summary))
        with contextlib.suppress(Exception):
            await self.strategy.aclose()
        self._broadcast({"type": "status", **self.info()})
        self._broadcast({"type": "summary", "summary": summary})

    async def _run_realtime(self) -> None:
        dt = self.engine.dt
        frame_interval = 1.0 / self._fps
        last_wall = time.monotonic()
        last_frame = 0.0
        last_flush = last_wall
        debt = 0.0
        while self._active():
            if not self._resume.is_set():
                await self._resume.wait()
                last_wall = time.monotonic()
            now = time.monotonic()
            debt += (now - last_wall) * self._speed
            last_wall = now
            steps = min(int(debt / dt), MAX_STEPS_PER_ITERATION)
            for _ in range(steps):
                self._tick()
                self._collect_decision()
                if self._decision_due():
                    self._launch_decision()
                if not self._active():
                    break
            debt = 0.0 if steps == MAX_STEPS_PER_ITERATION else debt - steps * dt
            if now - last_frame >= frame_interval:
                last_frame = now
                self._broadcast(self.frame())
            if now - last_flush >= FLUSH_EVERY_S:
                last_flush = now
                await self._flush()
            await asyncio.sleep(0.004)
        self._broadcast(self.frame())

    async def _run_headless(self) -> None:
        ticks = 0
        while self._active():
            self._tick()
            if self._decision_due():
                context, serial = self._context()
                decision = await self._safe_decide(context)
                self._apply(decision, context, serial)
            ticks += 1
            if ticks % 2000 == 0:
                await self._flush()
                await asyncio.sleep(0)

    # ================================================================== pipeline
    def _active(self) -> bool:
        return not self.engine.finished and not self._stop

    def _tick(self) -> None:
        engine = self.engine
        emergency = self.aggregator.emergency_detection()
        engine.signal.set_preemption(emergency[0] if emergency else None)
        engine.step(self.aggregator.has_conflicting_call())
        self.detectors.sample()
        self.aggregator.update()
        second = round(engine.time * 10) // 10
        if second != self._last_second:
            self._last_second = second
            sample = self.evaluator.sample()
            self.latest_sample = sample
            self._samples_buffer.append(sample.model_dump())

    def _decision_due(self) -> bool:
        return (
            self._pending is None
            and self.engine.time >= self._next_epoch
            and needs_decision(self.engine.signal)
        )

    def _context(self) -> tuple[DecisionContext, int]:
        signal = self.engine.signal
        context = DecisionContext(self.aggregator.observe(), legal_actions(signal), self.engine.timing)
        return context, len(signal.phase_changes)

    def _launch_decision(self) -> None:
        context, serial = self._context()
        self._pending = asyncio.ensure_future(self._safe_decide(context))
        self._pending_meta = (context, serial)

    def _collect_decision(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        task, self._pending = self._pending, None
        assert self._pending_meta is not None  # noqa: S101 - set together with _pending
        context, serial = self._pending_meta
        self._apply(task.result(), context, serial)

    async def _safe_decide(self, context: DecisionContext) -> Decision:
        try:
            return await self.strategy.decide(context)
        except Exception as exc:  # a strategy bug must never take the intersection down
            log.exception("strategy %s failed", self.strategy.key)
            return Decision(ControlAction.KEEP_CURRENT_PHASE, DecisionSource.SAFETY, f"strategy error: {exc}")

    def _apply(self, decision: Decision, context: DecisionContext, serial: int) -> None:
        signal = self.engine.signal
        stale = serial != len(signal.phase_changes) or signal.interval is not Interval.GREEN
        applied = None if stale else vet(decision.action, signal)
        if applied is not None:
            signal.command(applied)
        self._next_epoch = self.engine.time + self.strategy.decision_interval_s
        self.decisions.append(decision)
        self._record(decision, context, applied, stale)

    def _record(
        self, decision: Decision, context: DecisionContext, applied: ControlAction | None, stale: bool
    ) -> None:
        phase = context.observation.signal.current_phase
        entry: dict[str, Any] = {
            "seq": len(self.decisions),
            "t": round(self.engine.time, 1),
            "decided_at": context.observation.time_seconds,
            "phase": phase.value,
            "green_elapsed": context.observation.signal.phase_elapsed_seconds,
            "action": action_label(decision.action, phase),
            "applied": action_label(applied, phase) if applied else ("stale" if stale else "rejected"),
            "source": decision.source.value,
            "rationale": decision.rationale,
            "confidence": decision.confidence,
            "probabilities": decision.probabilities,
            "latency_ms": round(decision.latency_ms, 1) if decision.latency_ms is not None else None,
            "legal": [action_label(a, phase) for a in context.legal_actions],
            "queues": {a.value: context.observation.approaches[a].estimated_queue_vehicles for a in Approach},
        }
        if decision.jev is not None:
            jev = decision.jev
            entry["jev"] = {
                "model": jev.model,
                "choice": jev.choice,
                "congestion_score": jev.congestion_score,
                "congestion_confidence": jev.congestion_confidence,
                "input_tokens": jev.input_tokens,
                "output_tokens": jev.output_tokens,
                "error": jev.error,
                "state": jev.state,
                "questions": jev.questions,
            }
        self.decision_log.append(entry)
        self._decisions_buffer.append(entry)
        self._broadcast({"type": "decision", "decision": entry})

    # ================================================================== streaming
    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self, message: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # drop the oldest frame: live view prefers freshness
            queue.put_nowait(message)

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario": self.scenario.key,
            "scenario_name": self.scenario.name,
            "controller": self.strategy.key,
            "controller_name": self.strategy.name,
            "seed": self.seed,
            "status": self.status.value,
            "speed": self._speed,
            "time": round(self.engine.time, 1),
            "duration": self.scenario.duration_s,
            "error": self.error,
        }

    def init_message(self) -> dict[str, Any]:
        g = self.engine.geometry
        detectors = []
        for det_id in self._detector_ids:
            loop = self.detectors.loops[det_id]
            x, y, heading = g.inbound_pose(loop.approach, loop.lane, np.array([loop.centre_m]))
            detectors.append(
                {
                    "id": det_id,
                    "role": loop.role.name.lower(),
                    "x": float(x[0]),
                    "y": float(y[0]),
                    "h": heading,
                }
            )
        return {
            "type": "init",
            "session": self.info(),
            "scenario": {"name": self.scenario.name, "tagline": self.scenario.tagline},
            "geometry": {
                "lane_width": g.lane_width_m,
                "lanes": INBOUND_LANES,
                "approach_length": g.approach_length_m,
                "exit_length": g.exit_length_m,
                "stop_line": g.stop_line_offset_m,
                "crosswalk_centre": g.crosswalk_centre_offset_m,
                "crosswalk_width": g.crosswalk_width_m,
            },
            "detectors": detectors,
            "decisions": list(self.decision_log)[-12:],
        }

    def frame(self) -> dict[str, Any]:
        engine = self.engine
        signal = engine.signal
        poses = engine.poses()
        flags = poses.braking.astype(int) + 2 * (poses.turn == -1) + 4 * (poses.turn == 1)
        vehicles = [
            [int(vid), round(float(x), 2), round(float(y), 2), round(float(h), 3), int(c), int(fl)]
            for vid, x, y, h, c, fl in zip(
                poses.vid, poses.x, poses.y, poses.heading, poses.cls, flags, strict=True
            )
        ]
        pedestrians = [[pid, round(x, 2), round(y, 2), c] for pid, x, y, c in engine.pedestrians.poses()]
        states = self.aggregator.detector_states()
        sample = self.latest_sample
        return {
            "type": "frame",
            "t": round(engine.time, 2),
            "status": self.status.value,
            "speed": self._speed,
            "signal": {
                "phase": signal.phase.value,
                "interval": signal.interval.value,
                "interval_elapsed": round(signal.interval_elapsed, 1),
                "green_elapsed": round(signal.green_elapsed, 1),
                "min_green": signal.min_green_effective,
                "max_green": signal.timing.max_green_s,
                "extension": round(signal.extension_remaining, 1),
                "colors": {a.value: signal.color(a).value for a in Approach},
                "ped": {a.value: signal.ped_signal(a).value for a in Approach},
                "preempt": signal.preempted_by.value if signal.preempted_by else None,
            },
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "detectors": "".join("1" if on else "0" for _, on, _ in states),
            "faults": [det_id for det_id, _, fault in states if fault],
            "queues": {
                "estimated": sample.estimated_queue if sample else {},
                "true": sample.true_queue if sample else {},
                "off_map": sample.off_map_queue if sample else {},
            },
            "kpi": sample.model_dump() if sample else None,
        }

    # ================================================================== persistence
    async def _flush(self) -> None:
        samples, self._samples_buffer = self._samples_buffer, []
        decisions, self._decisions_buffer = self._decisions_buffer, []
        if samples or decisions:
            await self._persist(
                lambda r: (r.add_samples(self.id, samples), r.add_decisions(self.id, decisions))
            )

    async def _persist(self, op: Any) -> None:
        if self._repo is None:
            return
        repo = self._repo
        try:
            await asyncio.to_thread(op, repo)
        except Exception:
            log.exception("persistence failed for session %s", self.id)
