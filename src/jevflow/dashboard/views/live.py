"""Live control room: run a scenario and watch the controller decide in real time."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from jevflow.config import get_settings
from jevflow.dashboard import charts
from jevflow.dashboard.api_client import ApiError
from jevflow.dashboard.components.intersection import live_intersection
from jevflow.dashboard.state import api, controllers, header, require_api, scenarios

SESSION_KEY = "live_session_id"

PIPELINE = (
    ("01", "Ground truth", "IDM car-following, heterogeneous drivers, gap acceptance, pedestrians, EVs."),
    ("02", "Virtual loops", "Stop-line, advance speed-trap and entry loops with misses, noise and faults."),
    ("03", "Aggregator", "Input–output queue counting, FIFO delay, speed, detector health checks."),
    ("04", "Jev judges", "Chooses among legal actions only, with probabilities and a confidence."),
    ("05", "Controller acts", "Min/max green, yellow, all-red, walk and preemption are enforced in code."),
)


def _sidebar(info: dict[str, Any]) -> None:
    scenario_list = scenarios()
    by_key = {s["key"]: s for s in scenario_list}
    ctrl_list = controllers()

    st.sidebar.markdown("### Run a simulation")
    scenario_key = st.sidebar.selectbox(
        "Scenario", list(by_key), format_func=lambda k: by_key[k]["name"], key="live_scenario"
    )
    st.sidebar.caption(by_key[scenario_key]["tagline"])
    available = [c for c in ctrl_list if c["available"]]
    labels = {c["key"]: c["name"] for c in ctrl_list}
    controller = st.sidebar.radio(
        "Controller", [c["key"] for c in available], format_func=lambda k: labels[k], key="live_controller"
    )
    if not info["jev_enabled"]:
        st.sidebar.info(
            "Set `TYPESAFE_API_KEY` and restart the API to enable **Jev**.", icon=":material/key:"
        )
    col1, col2 = st.sidebar.columns(2)
    seed = col1.number_input("Seed", min_value=0, max_value=99999, value=42, step=1, key="live_seed")
    speed = col2.select_slider("Speed", options=[1, 2, 4, 8, 16], value=4, key="live_speed")
    if st.sidebar.button(
        "Start simulation", type="primary", use_container_width=True, icon=":material/play_arrow:"
    ):
        try:
            session = api().create_session(scenario_key, controller, int(seed), float(speed))
            st.session_state[SESSION_KEY] = session["id"]
            st.rerun()
        except ApiError as exc:
            st.sidebar.error(str(exc))

    session_id = st.session_state.get(SESSION_KEY)
    if session_id:
        st.sidebar.divider()
        st.sidebar.markdown("### Current run")
        c1, c2, c3 = st.sidebar.columns(3)
        if c1.button("", help="Pause", use_container_width=True, icon=":material/pause:"):
            _control(session_id, action="pause")
        if c2.button("", help="Resume", use_container_width=True, icon=":material/play_arrow:"):
            _control(session_id, action="resume")
        if c3.button("", help="Stop", use_container_width=True, icon=":material/stop:"):
            _control(session_id, action="stop")
        new_speed = st.sidebar.select_slider(
            "Live speed", options=[0.5, 1, 2, 4, 8, 16, 32], value=float(speed), key="live_speed_adjust"
        )
        if st.sidebar.button("Apply speed", use_container_width=True):
            _control(session_id, speed=float(new_speed))


def _control(session_id: str, action: str | None = None, speed: float | None = None) -> None:
    try:
        api().control(session_id, action=action, speed=speed)
    except ApiError as exc:
        st.sidebar.error(str(exc))


def _empty_state() -> None:
    st.markdown('<div class="jf-pipeline">' + "".join(
        f'<div class="jf-step"><span class="n">{n}</span><b>{title}</b><small>{text}</small></div>'
        for n, title, text in PIPELINE
    ) + "</div>", unsafe_allow_html=True)  # fmt: skip
    st.info(
        "Pick a scenario and a controller in the sidebar, then press **Start simulation**.",
        icon=":material/traffic:",
    )


@st.fragment(run_every=3)
def _analytics(session_id: str) -> None:
    try:
        samples = api().session_metrics(session_id)
        decisions = api().session_decisions(session_id, limit=40)
    except ApiError as exc:
        st.warning(str(exc))
        return
    if len(samples) < 3:
        st.caption("Collecting data…")
        return
    st.markdown("#### Queue per approach · what the controller sees vs what is really there")
    st.plotly_chart(charts.queue_small_multiples(samples), use_container_width=True, key="queues")
    left, right = st.columns(2)
    left.plotly_chart(charts.delay_trend(samples), use_container_width=True, key="delay")
    right.plotly_chart(charts.throughput_trend(samples), use_container_width=True, key="throughput")
    if decisions:
        st.markdown("#### Decision log")
        frame = pd.DataFrame(
            [
                {
                    "t (s)": d["t"],
                    "phase": d["phase"].replace("_", "-"),
                    "green (s)": d["green_elapsed"],
                    "decision": d["action"],
                    "applied": d["applied"],
                    "source": d["source"],
                    "confidence": d["confidence"],
                    "latency (ms)": d["latency_ms"],
                    "why": d["rationale"],
                }
                for d in decisions
            ]
        )
        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
            height=320,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "why": st.column_config.TextColumn("why", width="large"),
            },
        )


def render() -> None:
    info = require_api()
    _sidebar(info)
    header("Jev<span>Flow</span> · live control room", "")
    session_id = st.session_state.get(SESSION_KEY)
    if not session_id:
        _empty_state()
        return
    try:
        api().session(session_id)
    except ApiError:
        st.session_state.pop(SESSION_KEY, None)
        st.warning("That session is no longer available (the API may have restarted).")
        _empty_state()
        return
    live_intersection(session_id, get_settings().ws_url, height=790)
    _analytics(session_id)
