from __future__ import annotations

import pytest

from jevflow.domain.enums import Approach, ControlAction, Interval, Phase, SignalColor
from jevflow.domain.timing import SignalTiming, ite_all_red_interval, ite_yellow_interval
from jevflow.simulation.signal import IllegalCommandError, PedSignal, SignalController

DT = 0.1


def advance(signal: SignalController, seconds: float, conflicting: bool = True) -> None:
    for _ in range(round(seconds / DT)):
        signal.step(DT, conflicting)


def test_ite_formulas_match_field_values() -> None:
    assert ite_yellow_interval(50 / 3.6) == pytest.approx(3.28, abs=0.01)
    assert ite_all_red_interval(50 / 3.6, 16.0) == pytest.approx(1.58, abs=0.01)


def test_min_green_is_enforced() -> None:
    signal = SignalController(SignalTiming(min_green_s=10))
    advance(signal, 5)
    with pytest.raises(IllegalCommandError):
        signal.command(ControlAction.SWITCH_PHASE)
    advance(signal, 5.1)
    signal.command(ControlAction.SWITCH_PHASE)
    assert signal.interval is Interval.YELLOW


def test_change_sequence_is_yellow_then_all_red_then_other_green() -> None:
    timing = SignalTiming(min_green_s=5, yellow_s=3.5, all_red_s=2.0)
    signal = SignalController(timing)
    advance(signal, 5.1)
    signal.command(ControlAction.SWITCH_PHASE)
    assert signal.color(Approach.NORTH) is SignalColor.YELLOW
    advance(signal, 3.6)
    assert signal.interval is Interval.ALL_RED
    assert all(c is SignalColor.RED for c in signal.colors().values())
    advance(signal, 2.1)
    assert signal.phase is Phase.EAST_WEST and signal.interval is Interval.GREEN


def test_max_out_only_with_a_conflicting_call() -> None:
    signal = SignalController(SignalTiming(max_green_s=30))
    advance(signal, 40, conflicting=False)
    assert signal.interval is Interval.GREEN, "rests in green without demand"
    advance(signal, 0.2, conflicting=True)
    assert signal.interval is Interval.YELLOW


def test_extension_cannot_exceed_max_green() -> None:
    signal = SignalController(SignalTiming(max_green_s=30, extension_s=5))
    advance(signal, 26, conflicting=False)
    assert not signal.can_extend()
    with pytest.raises(IllegalCommandError):
        signal.command(ControlAction.EXTEND_CURRENT_GREEN)


def test_pedestrian_call_lengthens_min_green_and_shows_walk() -> None:
    timing = SignalTiming(min_green_s=10, walk_s=7, ped_clearance_s=13)
    signal = SignalController(timing)
    signal.call_pedestrian(Approach.NORTH)  # crosswalk across the north leg walks with east-west
    advance(signal, 10.1)
    signal.command(ControlAction.SWITCH_PHASE)
    advance(signal, 3.6 + 2.1)
    assert signal.phase is Phase.EAST_WEST
    assert signal.ped_signal(Approach.NORTH) is PedSignal.WALK
    assert signal.min_green_effective == pytest.approx(20)
    advance(signal, 8)
    assert signal.ped_signal(Approach.NORTH) is PedSignal.FLASHING_DONT_WALK


def test_preemption_forces_switch_but_keeps_clearance_intervals() -> None:
    signal = SignalController(SignalTiming(min_green_s=10, preempt_min_green_s=5))
    signal.set_preemption(Approach.EAST)
    advance(signal, 5.1)
    assert signal.interval is Interval.YELLOW
    advance(signal, 3.6 + 2.1)
    assert signal.phase is Phase.EAST_WEST
    advance(signal, 120)
    assert signal.phase is Phase.EAST_WEST, "holds green while the emergency vehicle approaches"
    assert not signal.can_terminate()
