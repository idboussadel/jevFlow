"""Jev questions: the typed judgements we ask for, with domain knowledge in the criteria.

Criteria are structured objects (``what`` / ``prefer_when`` / ``avoid_when``), as TypeSafe
recommends when options are easy to confuse. Only *legal* actions are ever offered, so the
typed answer space itself guarantees Jev cannot choose something unsafe.
"""

from __future__ import annotations

from typing import Any

from typesafe_sdk import Choice, Score

from jevflow.control.jev.state import action_label
from jevflow.domain.enums import ControlAction, Phase

NEXT_ACTION = "next_action"
CONGESTION = "congestion"


# Match ActuatedStrategy's passage-time gap-out (~3 s without actuation on the green side).
PASSAGE_TIME_S = 3.0


def _criteria(action: ControlAction, current: Phase, extension_s: float) -> dict[str, Any]:
    served, waiting = current.label, current.other.label
    match action:
        case ControlAction.KEEP_CURRENT_PHASE:
            return {
                "what": f"Keep {served} green and re-evaluate in about a second.",
                "prefer_when": (
                    f"{served} still has recent detections (seconds_since_last_detection under "
                    f"{PASSAGE_TIME_S:.0f}s on a green approach), a queue still clearing "
                    f"(departures_30s > 0 with estimated_queue left), or {waiting} has no real call."
                ),
                "avoid_when": (
                    f"{served} has gapped out (no actuation for ~{PASSAGE_TIME_S:.0f}s+) and {waiting} "
                    "has a queue, spillback, pedestrians, an emergency vehicle, or a clearly worse wait."
                ),
            }
        case ControlAction.EXTEND_CURRENT_GREEN:
            return {
                "what": f"Commit to {extension_s:.0f} more seconds of {served} green before deciding again.",
                "prefer_when": (
                    f"A platoon or clearing queue on {served} is still actuating detectors; extending "
                    "avoids cutting mid-discharge. Use this instead of switching while the green side "
                    f"has not yet gapped out (~{PASSAGE_TIME_S:.0f}s quiet)."
                ),
                "avoid_when": (
                    f"{served} arrivals are sparse / already gapped out, or {waiting} has spillback, "
                    "pedestrians, an emergency vehicle, or an extreme red wait that outweighs finishing "
                    "the green discharge."
                ),
            }
        case ControlAction.SWITCH_PHASE:
            return {
                "what": f"End {served} green now (yellow, all-red) and give green to {waiting}.",
                "prefer_when": (
                    f"{served} has gapped out (~{PASSAGE_TIME_S:.0f}s+ without actuation) with little "
                    f"queue left, or {waiting} has spillback, an emergency vehicle, pedestrians waiting, "
                    f"or a clearly much worse wait — not merely a slightly larger queue."
                ),
                "avoid_when": (
                    f"{served} is still detecting arrivals or clearing a queue; every switch costs "
                    "yellow + all-red + start-up lost time and creates extra stops. Do not switch just "
                    f"because {waiting} queue is a bit larger."
                ),
            }


def next_action_question(legal: tuple[ControlAction, ...], current: Phase, extension_s: float) -> Choice:
    return Choice(
        instructions=(
            "You are an actuated-style adaptive controller. Prefer KEEP or EXTEND while the green "
            "side still has detections or a clearing queue. Prefer SWITCH mainly on gap-out "
            f"(~{PASSAGE_TIME_S:.0f}s without actuation on green), spillback, emergency, pedestrians, "
            "or a clearly worse red wait — not a slightly larger opposing queue. Minimise stops and "
            "lost time from unnecessary switches."
        ),
        criteria={action_label(a, current): _criteria(a, current, extension_s) for a in legal},
    )


def congestion_question() -> Score:
    return Score(
        instructions="How congested is this intersection right now, across all approaches?",
        criteria=[
            "Free flow: queues of a few vehicles at most, and every vehicle clears on its first green.",
            "Light: short queues that fully clear each cycle; delays are brief.",
            "Moderate: some queues of roughly 6 to 12 vehicles; a few vehicles wait more than one cycle.",
            "Heavy: long queues on at least one approach, many vehicles waiting through multiple cycles.",
            "Gridlock risk: queues are spilling back to upstream intersections or waits are extreme.",
        ],
    )
