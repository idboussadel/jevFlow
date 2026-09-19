from __future__ import annotations

from jevflow.control.base import DecisionContext
from jevflow.control.baselines import ActuatedStrategy, FixedTimeStrategy, MaxPressureStrategy
from jevflow.domain.enums import ControlAction
from jevflow.simulation.engine import resolve_timing
from jevflow.simulation.scenarios import get_scenario


def test_webster_plan_gives_the_major_street_more_green() -> None:
    scenario = get_scenario("rush_hour")
    cycle, greens = FixedTimeStrategy.plan(scenario, resolve_timing(scenario))
    assert 50 <= cycle <= 150
    assert greens[max(greens, key=greens.__getitem__)].real >= min(greens.values())
    assert greens[next(iter(greens))] > greens[list(greens)[1]]  # north-south dominates


async def test_all_baselines_return_legal_actions(decision_context: DecisionContext) -> None:
    for strategy in (
        ActuatedStrategy(),
        MaxPressureStrategy(),
        FixedTimeStrategy(get_scenario("rush_hour"), decision_context.timing),
    ):
        decision = await strategy.decide(decision_context)
        assert decision.action in decision_context.legal_actions
        assert decision.rationale


async def test_actuated_rests_in_green_without_conflicting_demand(decision_context: DecisionContext) -> None:
    obs = decision_context.observation
    other = obs.signal.current_phase.other
    quiet = {
        a: m.model_copy(update={"estimated_queue_vehicles": 0.0, "stop_line_occupancy_percent": 0.0})
        if a.phase is other
        else m
        for a, m in obs.approaches.items()
    }
    calls = dict.fromkeys(obs.pedestrian_calls, False)
    context = DecisionContext(
        obs.model_copy(update={"approaches": quiet, "pedestrian_calls": calls}),
        decision_context.legal_actions,
        decision_context.timing,
    )
    decision = await ActuatedStrategy().decide(context)
    assert decision.action is ControlAction.KEEP_CURRENT_PHASE
