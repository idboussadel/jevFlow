"""Jev as a traffic-signal controller.

Control flow follows TypeSafe's recommended *Observe → Judge → Act* shape:

1. **Observe**: the detector layer produces a :class:`TrafficObservation`; the safety guard
   derives the legal actions.
2. **Judge**: Jev picks one legal action and returns calibrated probabilities and a confidence.
3. **Act**: Jev's choice is applied whenever a usable answer arrives. Only API errors and an
   open circuit breaker hand control to the actuated fallback, so the intersection never
   waits on the network.
"""

from __future__ import annotations

import time
from typing import ClassVar

from typesafe_sdk import TypeSafeError

from jevflow.control.base import ControlStrategy, Decision, DecisionContext, JevTrace
from jevflow.control.baselines import ActuatedStrategy
from jevflow.control.jev.client import DecisionModel
from jevflow.control.jev.questions import CONGESTION, NEXT_ACTION, congestion_question, next_action_question
from jevflow.control.jev.state import action_label, build_state
from jevflow.domain.enums import ControlAction, DecisionSource


class CircuitBreaker:
    """Stops calling a failing dependency for a cool-down period (wall-clock)."""

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 20.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            self._opened_at = None  # half-open: allow a trial call
            self._failures = self.failure_threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class JevStrategy(ControlStrategy):
    key: ClassVar[str] = "jev"
    name: ClassVar[str] = "Jev (TypeSafe)"
    description: ClassVar[str] = (
        "Jev chooses among legal actions from detector data; actuated only if Jev is unavailable."
    )

    def __init__(
        self,
        model: DecisionModel,
        *,
        decision_interval_s: float = 1.0,
        ask_congestion: bool = True,
        fallback: ControlStrategy | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._model = model
        self.decision_interval_s = decision_interval_s
        self.ask_congestion = ask_congestion
        self._fallback = fallback or ActuatedStrategy()
        self._breaker = breaker or CircuitBreaker()

    async def decide(self, context: DecisionContext) -> Decision:
        if self._breaker.is_open:
            return await self._fall_back(context, "Jev circuit open after repeated failures", trace=None)

        current = context.observation.signal.current_phase
        state = build_state(context).model_dump(mode="json")
        questions = {
            NEXT_ACTION: next_action_question(context.legal_actions, current, context.timing.extension_s)
        }
        if self.ask_congestion:
            questions[CONGESTION] = congestion_question()  # type: ignore[assignment]
        serialized = {name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()}

        started = time.perf_counter()
        try:
            response = await self._model.evaluate(state, questions)
        except TypeSafeError as exc:
            self._breaker.record_failure()
            trace = JevTrace(self._model.model, state, serialized, error=f"{type(exc).__name__}: {exc}")
            return await self._fall_back(context, f"Jev unavailable ({type(exc).__name__})", trace=trace)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._breaker.record_success()

        answer = response.choices.get(NEXT_ACTION)
        congestion = response.scores.get(CONGESTION)
        trace = JevTrace(
            model=response.model,
            state=state,
            questions=serialized,
            choice=answer.choice if answer else None,
            probabilities=dict(answer.probabilities) if answer else {},
            confidence=answer.confidence if answer else None,
            congestion_score=congestion.score if congestion else None,
            congestion_confidence=congestion.confidence if congestion else None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        labels = {action_label(a, current): a for a in context.legal_actions}
        if answer is None or answer.choice not in labels:
            return await self._fall_back(
                context, "Jev returned no usable choice", trace=trace, latency_ms=latency_ms
            )

        top = answer.probabilities.get(answer.choice, 0.0)
        return Decision(
            action=labels[answer.choice],
            source=DecisionSource.JEV,
            rationale=f"Jev chose {answer.choice} (p={top:.2f}, confidence {answer.confidence:.2f})",
            latency_ms=latency_ms,
            probabilities=dict(answer.probabilities),
            confidence=answer.confidence,
            jev=trace,
        )

    async def _fall_back(
        self,
        context: DecisionContext,
        reason: str,
        *,
        trace: JevTrace | None,
        latency_ms: float | None = None,
    ) -> Decision:
        fallback = await self._fallback.decide(context)
        action = (
            fallback.action if fallback.action in context.legal_actions else ControlAction.KEEP_CURRENT_PHASE
        )
        return Decision(
            action=action,
            source=DecisionSource.FALLBACK,
            rationale=f"{reason}; actuated fallback: {fallback.rationale}",
            latency_ms=latency_ms,
            probabilities=dict(trace.probabilities) if trace and trace.probabilities else None,
            confidence=trace.confidence if trace else None,
            jev=trace,
        )

    async def aclose(self) -> None:
        await self._model.aclose()
