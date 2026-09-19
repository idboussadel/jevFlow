"""Built-in scenario library.

Demand levels are chosen with a capacity check in mind: a shared through/right lane
discharges at roughly 1,800 veh/h of green, so the critical flow ratios below range from
comfortable (late night) to oversaturated (stadium surge).
"""

from __future__ import annotations

from jevflow.domain.enums import Approach
from jevflow.simulation.scenario import (
    ApproachDemand,
    DemandPoint,
    EmergencyEvent,
    FaultEvent,
    Scenario,
    SensorProfile,
    TurningShares,
)

N, S, E, W = Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST


def _profile(*points: tuple[float, float]) -> tuple[DemandPoint, ...]:
    return tuple(DemandPoint(at_s=t, vph=q) for t, q in points)


def _flat(vph: float) -> tuple[DemandPoint, ...]:
    return _profile((0, vph))


def _peds(rate: float) -> dict[Approach, float]:
    return dict.fromkeys(Approach, rate)


RUSH_HOUR = Scenario(
    key="rush_hour",
    name="Morning Rush",
    tagline="A north–south commuter wave builds, peaks, and fades",
    description=(
        "North–south is the commuter corridor: demand ramps from 500 to 950 veh/h per approach and "
        "arrives in platoons released by the upstream signal. East–west is a side street. A good "
        "controller gives the corridor long greens without starving the side street."
    ),
    demand={
        N: ApproachDemand(
            profile=_profile((0, 520), (300, 950), (660, 900), (900, 650)), platoon_strength=0.45
        ),
        S: ApproachDemand(
            profile=_profile((0, 460), (300, 820), (660, 800), (900, 560)), platoon_strength=0.4
        ),
        E: ApproachDemand(profile=_profile((0, 260), (450, 380), (900, 300))),
        W: ApproachDemand(profile=_profile((0, 300), (450, 420), (900, 320))),
    },
    pedestrians_per_hour=_peds(45),
)

BALANCED = Scenario(
    key="balanced_midday",
    name="Balanced Midday",
    tagline="Even demand, lots of pedestrians, gentle platoons",
    description=(
        "Every approach carries about 480 veh/h with moderate platooning and 90 pedestrians per hour "
        "per crosswalk. Pedestrian calls lengthen minimum greens, which penalises frequent switching."
    ),
    demand={a: ApproachDemand(profile=_flat(480), platoon_strength=0.3) for a in Approach},
    pedestrians_per_hour=_peds(90),
)

STADIUM_SURGE = Scenario(
    key="stadium_surge",
    name="Stadium Let-Out",
    tagline="East–west demand surges past capacity; short approaches spill back",
    description=(
        "Ten minutes after kick-off ends, east–west demand jumps from 350 to 880 veh/h while "
        "north–south stays busy. Approaches are only 180 m long, so queues reach the upstream "
        "junction and vehicles wait off-map. Managing spillback is the whole game."
    ),
    approach_length_m=180,
    demand={
        N: ApproachDemand(profile=_flat(600), platoon_strength=0.25),
        S: ApproachDemand(profile=_flat(560), platoon_strength=0.25),
        E: ApproachDemand(profile=_profile((0, 350), (150, 880), (600, 860), (780, 420))),
        W: ApproachDemand(profile=_profile((0, 330), (150, 820), (600, 800), (780, 380))),
    },
    pedestrians_per_hour=_peds(30),
)

EMERGENCY = Scenario(
    key="emergency_corridor",
    name="Emergency Response",
    tagline="Three ambulances cross a busy junction; preemption must clear the way",
    description=(
        "Busy two-way traffic with ambulances arriving from the east, north and west. Their "
        "transponders trigger preemption in the signal controller. That is a deterministic safety "
        "function, never delegated to the AI, and the controller has to recover the queues it causes."
    ),
    demand={
        N: ApproachDemand(profile=_flat(640), platoon_strength=0.3),
        S: ApproachDemand(profile=_flat(600), platoon_strength=0.3),
        E: ApproachDemand(profile=_flat(420)),
        W: ApproachDemand(profile=_flat(400)),
    },
    emergencies=(
        EmergencyEvent(at_s=150, approach=E),
        EmergencyEvent(at_s=430, approach=N),
        EmergencyEvent(at_s=700, approach=W),
    ),
    pedestrians_per_hour=_peds(40),
)

SENSOR_FAULTS = Scenario(
    key="sensor_faults",
    name="Failing Detectors",
    tagline="Stuck, dead and chattering loops: can control degrade gracefully?",
    description=(
        "Rush-hour demand with worn-out detection. Loops miss 3% of vehicles, the north advance "
        "loop sticks ON, the east stop-line loop goes dead, and a west loop chatters. The aggregator "
        "flags each fault and falls back to the remaining detectors. Controllers only see the "
        "degraded picture."
    ),
    demand=RUSH_HOUR.demand,
    pedestrians_per_hour=_peds(30),
    sensors=SensorProfile(
        miss_rate=0.03,
        false_actuations_per_hour=6,
        faults=(
            FaultEvent(approach=N, lane=1, detector="advance", mode="stuck_on", start_s=120, duration_s=360),
            FaultEvent(approach=E, lane=1, detector="stop", mode="dead", start_s=200, duration_s=500),
            FaultEvent(approach=W, lane=1, detector="advance", mode="chatter", start_s=380, duration_s=240),
        ),
    ),
)

LEFT_TURN_CRUNCH = Scenario(
    key="left_turn_crunch",
    name="Left-Turn Crunch",
    tagline="Heavy permissive lefts against strong opposing flow",
    description=(
        "A third of north–south drivers turn left, and they must find gaps in the opposing stream. "
        "Long greens help the throughs but starve the lefts until the yellow. The left-turn "
        "queue is visible only through its own lane's detectors."
    ),
    demand={
        N: ApproachDemand(profile=_flat(700), turning=TurningShares(left=0.32, through=0.58, right=0.10)),
        S: ApproachDemand(profile=_flat(680), turning=TurningShares(left=0.30, through=0.60, right=0.10)),
        E: ApproachDemand(profile=_flat(380)),
        W: ApproachDemand(profile=_flat(360)),
    },
    pedestrians_per_hour=_peds(25),
)

LATE_NIGHT = Scenario(
    key="late_night",
    name="Late Night",
    tagline="Sparse traffic: nobody should wait at an empty junction",
    description=(
        "Light, random traffic of 120 to 220 veh/h. Fixed-time plans make drivers sit at red in front "
        "of an empty cross street; responsive control should rest in green and switch on demand."
    ),
    demand={
        N: ApproachDemand(profile=_flat(220)),
        S: ApproachDemand(profile=_flat(190)),
        E: ApproachDemand(profile=_flat(140)),
        W: ApproachDemand(profile=_flat(120)),
    },
    pedestrians_per_hour=_peds(8),
)

SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in (RUSH_HOUR, BALANCED, STADIUM_SURGE, EMERGENCY, SENSOR_FAULTS, LEFT_TURN_CRUNCH, LATE_NIGHT)
}


def get_scenario(key: str) -> Scenario:
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(f"unknown scenario {key!r}; choose from {sorted(SCENARIOS)}") from None
