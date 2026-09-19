"""Strategy registry and factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jevflow.config import Settings
from jevflow.control.base import ControlStrategy
from jevflow.control.baselines import ActuatedStrategy, FixedTimeStrategy, MaxPressureStrategy
from jevflow.control.jev.client import TypeSafeDecisionModel
from jevflow.control.jev.strategy import JevStrategy
from jevflow.domain.timing import SignalTiming
from jevflow.simulation.scenario import Scenario


class StrategyUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StrategyInfo:
    key: str
    name: str
    description: str
    available: bool
    reason: str | None = None


Builder = Callable[[Scenario, SignalTiming, Settings], ControlStrategy]


def _jev(_: Scenario, __: SignalTiming, settings: Settings) -> ControlStrategy:
    if not settings.jev_enabled or settings.typesafe_api_key is None:
        raise StrategyUnavailableError("Set TYPESAFE_API_KEY to enable Jev (https://console.typesafe.ai/)")
    model = TypeSafeDecisionModel(
        api_key=settings.typesafe_api_key.get_secret_value(),
        model=settings.jev_model,
        timeout_s=settings.jev_timeout_s,
        max_retries=settings.jev_max_retries,
    )
    return JevStrategy(
        model,
        decision_interval_s=settings.jev_decision_interval_s,
        ask_congestion=settings.jev_ask_congestion,
    )


_REGISTRY: dict[str, tuple[type[ControlStrategy], Builder]] = {
    JevStrategy.key: (JevStrategy, _jev),
    ActuatedStrategy.key: (ActuatedStrategy, lambda _s, _t, _c: ActuatedStrategy()),
    MaxPressureStrategy.key: (MaxPressureStrategy, lambda _s, _t, _c: MaxPressureStrategy()),
    FixedTimeStrategy.key: (FixedTimeStrategy, lambda s, t, _c: FixedTimeStrategy(s, t)),
}


def strategy_catalog(settings: Settings) -> list[StrategyInfo]:
    out: list[StrategyInfo] = []
    for key, (cls, _) in _REGISTRY.items():
        available = key != JevStrategy.key or settings.jev_enabled
        reason = None if available else "TYPESAFE_API_KEY is not set"
        out.append(StrategyInfo(key, cls.name, cls.description, available, reason))
    return out


def create_strategy(
    key: str, scenario: Scenario, timing: SignalTiming, settings: Settings
) -> ControlStrategy:
    try:
        _, build = _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown controller {key!r}; choose from {sorted(_REGISTRY)}") from None
    return build(scenario, timing, settings)
