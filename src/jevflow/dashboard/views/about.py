"""How it works: architecture and the realism behind the simulation."""

from __future__ import annotations

import streamlit as st

from jevflow.dashboard.state import header

ARCHITECTURE = """
digraph G {
  bgcolor="transparent"; rankdir=LR; pad=0.3; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, color="#2a2f3d",
        fillcolor="#151923", fontcolor="#e8e9ee", margin="0.18,0.1"];
  edge [color="#5b6070", arrowsize=0.7, fontname="Inter", fontsize=9, fontcolor="#7c7f8c"];
  truth [label="Simulation truth\\nIDM · drivers · peds · EVs"];
  loops [label="Virtual detectors\\nstop · advance · entry loops\\nmisses · noise · faults"];
  agg [label="Traffic-state aggregator\\nqueues · delay · speed · health"];
  guard [label="Safety guard\\nlegal actions"];
  jev [label="Jev (TypeSafe)\\nChoice + confidence", fillcolor="#16263f", color="#3987e5"];
  fb [label="Actuated fallback"];
  cab [label="Signal controller\\nmin/max green · yellow\\nall-red · walk · preempt"];
  eval [label="Ground-truth evaluator\\nHCM delay · LOS · fairness"];
  truth -> loops [label="presence"]; loops -> agg [label="actuations"];
  agg -> guard; guard -> jev [label="state + legal"]; jev -> cab [label="always"];
  jev -> fb [label="API error / circuit", style=dashed]; fb -> cab;
  cab -> truth [label="lights"]; truth -> eval [style=dotted];
}
"""

SECTIONS = (
    (
        "The simulator knows everything. The controller doesn't.",
        "Vehicles are simulated with the Intelligent Driver Model. Each driver has their own desired speed, "
        "time gap, acceleration, reaction time, gap acceptance and yellow-light behaviour. The controller "
        "never sees that. It gets actuations from inductive loops sampled at 10 Hz, with ~1–3% missed "
        "vehicles, false pulses, speed quantisation and scheduled faults (stuck-on, dead, chattering).",
    ),
    (
        "Queues are estimated the way agencies do it",
        "Input–output counting between an upstream entry loop and the stop-line loop, corrected by a "
        "start-up discharge-wave model and re-anchored whenever the queue fully clears. When a loop "
        "fails, the lane falls back to a historical-volume model and is flagged as degraded.",
    ),
    (
        "Jev judges; code enforces safety",
        "Following TypeSafe's guidance (“code calculates, Jev judges”), deterministic code computes the "
        "legal actions and precomputes phase-level summaries. Jev only ever sees legal options as its "
        "answer space. Minimum green, pedestrian clearance, yellow, all-red, max-out and emergency "
        "preemption live in the signal controller and can't be overridden.",
    ),
    (
        "Honest evaluation",
        "Controllers are graded on ground truth with HCM control delay and level of service, stops, "
        "throughput, fairness (Jain's index), spillback, red-light running and pedestrian waits. Vehicles "
        "still waiting at the end count with the delay accrued so far, so starving an approach can't win. "
        "Benchmarks use common random numbers: identical traffic for every controller.",
    ),
    (
        "Real-time, including latency",
        "In live mode Jev is called asynchronously: the intersection keeps moving while the answer is in "
        "flight. An answer that arrives after the phase has changed is rejected as stale. API errors and a "
        "circuit breaker hand control to an actuated fallback so the intersection never waits on the network.",
    ),
)


def render() -> None:
    header("How it works", "From vehicle physics to a typed, auditable AI decision, every second.")
    st.graphviz_chart(ARCHITECTURE, use_container_width=True)
    cols = st.columns(2)
    for i, (title, body) in enumerate(SECTIONS):
        cols[i % 2].markdown(
            f'<div class="jf-card" style="margin-bottom:12px"><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("#### References")
    st.markdown(
        "- Treiber, Hennecke & Helbing (2000). *Congested traffic states in empirical observations and microscopic simulations* (IDM).\n"
        "- Transportation Research Board. *Highway Capacity Manual*, 6th/7th ed. (control delay, LOS, saturation flow).\n"
        "- ITE (2020). *Guidelines for Determining Traffic Signal Change and Clearance Intervals*.\n"
        "- Varaiya (2013). *Max pressure control of a network of signalized intersections*.\n"
        "- Akçelik & Chung (1994). *Calibration of the bunched exponential distribution of arrival headways*.\n"
        "- Liu et al. (2009). *Real-time queue length estimation for congested signalized intersections*.\n"
        "- TypeSafe AI docs: [System One / Jev](https://docs.typesafe.ai/)."
    )
