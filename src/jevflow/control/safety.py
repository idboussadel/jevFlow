"""Safety guard: derives the legal action set and vets decisions before they reach the lights.

This is defence in depth. The signal controller already refuses unsafe commands. The guard
additionally (a) tells strategies up front which actions are legal, so Jev is only ever
offered safe choices, and (b) re-validates a decision at the moment it is applied, because
with asynchronous AI calls the world may have moved on while the answer was in flight.
"""

from __future__ import annotations

from jevflow.domain.enums import ControlAction, Interval
from jevflow.simulation.signal import SignalController


def legal_actions(signal: SignalController) -> tuple[ControlAction, ...]:
    if signal.interval is not Interval.GREEN:
        return ()
    actions = [ControlAction.KEEP_CURRENT_PHASE]
    if signal.can_extend():
        actions.append(ControlAction.EXTEND_CURRENT_GREEN)
    if signal.can_terminate():
        actions.append(ControlAction.SWITCH_PHASE)
    return tuple(actions)


def needs_decision(signal: SignalController) -> bool:
    """A strategy is consulted only when there is a real choice to make."""
    return (
        signal.interval is Interval.GREEN
        and signal.extension_remaining == 0.0
        and ControlAction.SWITCH_PHASE in legal_actions(signal)
    )


def vet(action: ControlAction, signal: SignalController) -> ControlAction | None:
    """Return the action if still legal now, otherwise ``None``."""
    return action if action in legal_actions(signal) else None
