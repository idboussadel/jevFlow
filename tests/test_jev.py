from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx2
import pytest
from typesafe_sdk import Choice, Question, SystemOneResponse, TypeSafeAPIConnectionError

from jevflow.control.base import DecisionContext
from jevflow.control.jev.client import TypeSafeDecisionModel
from jevflow.control.jev.questions import CONGESTION, NEXT_ACTION
from jevflow.control.jev.state import action_label, build_state
from jevflow.control.jev.strategy import CircuitBreaker, JevStrategy
from jevflow.domain.enums import ControlAction, DecisionSource


def _response(choice: str, probabilities: dict[str, float], confidence: float) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            NEXT_ACTION: {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": confidence},
            CONGESTION: {
                "type": "score", "score": 2.4, "confidence": 0.7,
                "legend": {str(i): f"level {i}" for i in range(5)},
                "probabilities": {"0": 0.05, "1": 0.1, "2": 0.3, "3": 0.45, "4": 0.1},
            },
        },
        "usage": {"input_tokens": 950, "output_tokens": 12},
    }  # fmt: skip


class FakeModel:
    model = "jev-latest"

    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self.payload = payload
        self.calls: list[tuple[Mapping[str, Any], Mapping[str, Question]]] = []

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        self.calls.append((state, questions))
        if isinstance(self.payload, Exception):
            raise self.payload
        return SystemOneResponse.model_validate_json(json.dumps(self.payload))  # as the SDK parses the wire

    async def aclose(self) -> None:
        pass


def _switch_label(ctx: DecisionContext) -> str:
    return action_label(ControlAction.SWITCH_PHASE, ctx.observation.signal.current_phase)


def test_state_matches_the_documented_shape(decision_context: DecisionContext) -> None:
    state = build_state(decision_context).model_dump(mode="json")
    assert state["signal"]["current_phase"].endswith("_green")
    assert set(state["approaches"]) == {"north", "south", "east", "west"}
    assert "keep_current_phase" in state["legal_actions"]
    assert _switch_label(decision_context) in state["legal_actions"]
    assert json.dumps(state)  # plain JSON, no enums or numpy types


async def test_confident_answer_is_applied(decision_context: DecisionContext) -> None:
    label = _switch_label(decision_context)
    probs = {label: 0.82, "keep_current_phase": 0.12, "extend_current_green": 0.06}
    model = FakeModel(_response(label, probs, 0.71))
    decision = await JevStrategy(model).decide(decision_context)
    assert decision.action is ControlAction.SWITCH_PHASE
    assert decision.source is DecisionSource.JEV
    assert decision.jev is not None and decision.jev.congestion_score == pytest.approx(2.4)
    _, questions = model.calls[0]
    question = questions[NEXT_ACTION]
    assert isinstance(question, Choice)
    offered = set(question.criteria)
    assert offered == {
        action_label(a, decision_context.observation.signal.current_phase)
        for a in decision_context.legal_actions
    }


async def test_low_confidence_answer_is_still_applied(decision_context: DecisionContext) -> None:
    label = _switch_label(decision_context)
    model = FakeModel(
        _response(label, {label: 0.4, "keep_current_phase": 0.35, "extend_current_green": 0.25}, 0.05)
    )
    decision = await JevStrategy(model).decide(decision_context)
    assert decision.source is DecisionSource.JEV
    assert decision.action is ControlAction.SWITCH_PHASE
    assert decision.confidence == pytest.approx(0.05)

async def test_api_failure_falls_back_and_opens_the_circuit(decision_context: DecisionContext) -> None:
    model = FakeModel(TypeSafeAPIConnectionError("network down"))
    strategy = JevStrategy(model, breaker=CircuitBreaker(failure_threshold=2, cooldown_s=60))
    for _ in range(3):
        decision = await strategy.decide(decision_context)
        assert decision.source is DecisionSource.FALLBACK
    assert len(model.calls) == 2, "third call short-circuits without hitting the API"


async def test_real_sdk_round_trip_with_mock_transport(decision_context: DecisionContext) -> None:
    """Exercise the actual typesafe-sdk serialisation and parsing against a mocked HTTP layer."""
    label = _switch_label(decision_context)
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        probs = {label: 0.2, "keep_current_phase": 0.7, "extend_current_green": 0.1}
        return httpx2.Response(200, json=_response("keep_current_phase", probs, 0.55))

    model = TypeSafeDecisionModel("test-key", "jev-latest", 2.0, 0, transport=httpx2.MockTransport(handler))
    decision = await JevStrategy(model).decide(decision_context)
    await model.aclose()
    assert seen["url"].endswith("/v1/systemone")
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "jev-latest"
    assert body["questions"][NEXT_ACTION]["type"] == "choice"
    assert body["questions"][CONGESTION]["type"] == "score"
    assert body["state"]["legal_actions"] == build_state(decision_context).legal_actions
    assert decision.action is ControlAction.KEEP_CURRENT_PHASE
    assert decision.confidence == pytest.approx(0.55)
