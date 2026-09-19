"""The contract every signal-control strategy implements (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from jevflow.domain.enums import ControlAction, DecisionSource
from jevflow.domain.observation import TrafficObservation
from jevflow.domain.timing import SignalTiming


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Everything a strategy may use: detector-derived observation plus the legal action set."""

    observation: TrafficObservation
    legal_actions: tuple[ControlAction, ...]
    timing: SignalTiming


@dataclass(frozen=True, slots=True)
class JevTrace:
    """Full audit trail of one Jev call, so every AI decision can be inspected and replayed."""

    model: str
    state: dict[str, Any]
    questions: dict[str, Any]
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    congestion_score: float | None = None
    congestion_confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Decision:
    action: ControlAction
    source: DecisionSource
    rationale: str
    latency_ms: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    jev: JevTrace | None = None


class ControlStrategy(ABC):
    """A pluggable decision policy. Strategies advise; the signal controller enforces safety."""

    key: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    decision_interval_s: float = 1.0

    @abstractmethod
    async def decide(self, context: DecisionContext) -> Decision: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources such as HTTP clients."""
