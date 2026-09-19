"""Benchmark: every controller, same scenario, same seed (common random numbers)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from jevflow.dashboard import charts
from jevflow.dashboard.api_client import ApiError
from jevflow.dashboard.state import (
    api,
    controller_chip,
    controllers,
    header,
    los_chip,
    require_api,
    scenarios,
)
from jevflow.dashboard.theme import CONTROLLER_LABELS

JOB_KEY = "benchmark_job_id"


def _form(info: dict[str, Any]) -> None:
    scenario_list = scenarios()
    by_key = {s["key"]: s for s in scenario_list}
    ctrl = controllers()
    with st.form("benchmark"):
        c1, c2, c3, c4 = st.columns([2.2, 3, 1, 1.3])
        scenario_key = c1.selectbox("Scenario", list(by_key), format_func=lambda k: by_key[k]["name"])
        options = [c["key"] for c in ctrl if c["available"]]
        chosen = c2.multiselect(
            "Controllers", options, default=options, format_func=lambda k: CONTROLLER_LABELS.get(k, k)
        )
        seed = c3.number_input("Seed", min_value=0, max_value=99999, value=42)
        duration = c4.select_slider(
            "Duration", options=[300, 600, 900, 1800], value=900, format_func=lambda s: f"{s // 60} min"
        )
        submitted = st.form_submit_button("Run benchmark", type="primary", icon=":material/speed:")
    if not info["jev_enabled"]:
        st.caption("Jev is disabled because `TYPESAFE_API_KEY` is not set, so only the baselines will run.")
    if submitted:
        if not chosen:
            st.warning("Choose at least one controller.")
            return
        try:
            job = api().start_benchmark(scenario_key, chosen, int(seed), float(duration))
            st.session_state[JOB_KEY] = job["id"]
        except ApiError as exc:
            st.error(str(exc))


def _wait(job_id: str) -> dict[str, Any]:
    job = api().benchmark(job_id)
    if job["status"] != "running":
        return job
    with st.status("Running headless simulations…", expanded=True) as status:
        progress = st.empty()
        while job["status"] == "running":
            progress.markdown(
                "  \n".join(
                    f"{controller_chip(c)} &nbsp; `{job['progress'].get(c, 'queued')}`"
                    for c in job["controllers"]
                ),
                unsafe_allow_html=True,
            )
            time.sleep(1.0)
            job = api().benchmark(job_id)
        status.update(label="Benchmark complete", state="complete", expanded=False)
    return job


def _headline(results: dict[str, dict[str, Any]]) -> None:
    ok = {c: r["summary"] for c, r in results.items() if r.get("summary")}
    if not ok:
        return
    best = min(ok, key=lambda c: ok[c]["avg_control_delay_s"])
    cols = st.columns(len(ok))
    for col, (c, s) in zip(cols, ok.items(), strict=True):
        with col:
            st.markdown(controller_chip(c) + " " + los_chip(s["level_of_service"]), unsafe_allow_html=True)
            delta = None
            if "jev" in ok and c != "jev":
                ref = ok["jev"]["avg_control_delay_s"]
                delta = f"{(s['avg_control_delay_s'] - ref) / max(ref, 1e-6) * 100:+.0f}% vs Jev"
            st.metric("Avg control delay", f"{s['avg_control_delay_s']:.1f} s", delta, delta_color="off")
    if "jev" in ok and len(ok) > 1:
        baselines = {c: v for c, v in ok.items() if c != "jev"}
        strongest = min(baselines, key=lambda c: baselines[c]["avg_control_delay_s"])
        jev, base = ok["jev"]["avg_control_delay_s"], baselines[strongest]["avg_control_delay_s"]
        change = (jev - base) / max(base, 1e-6) * 100
        verdict = "less" if change < 0 else "more"
        st.markdown(
            f'<div class="jf-card" style="margin-top:12px"><span class="jf-big">{abs(change):.0f}%</span>'
            f"<p>{verdict} average delay with <b>Jev</b> than the strongest baseline "
            f"({CONTROLLER_LABELS[strongest]}) on identical traffic.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Lowest average delay: **{CONTROLLER_LABELS.get(best, best)}**.")


def _table(results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for c, r in results.items():
        s = r.get("summary")
        if not s:
            rows.append({"controller": CONTROLLER_LABELS.get(c, c), "error": r.get("error")})
            continue
        rows.append(
            {
                "controller": CONTROLLER_LABELS.get(c, c),
                "LOS": s["level_of_service"],
                "avg delay (s)": s["avg_control_delay_s"],
                "p95 delay (s)": s["p95_delay_s"],
                "stops/veh": s["stops_per_vehicle"],
                "throughput (veh/h)": s["throughput_vph"],
                "longest wait (s)": s["max_wait_s"],
                "fairness": s["fairness_index"],
                "spillback (s)": s["spillback_seconds"],
                "red-light runs": s["red_light_violations"],
                "ped wait (s)": s["pedestrian_avg_wait_s"],
                "EV delay (s)": s["emergency_avg_delay_s"],
                "switches": s["phase_switches"],
                "Jev calls": s["jev_calls"],
                "Jev latency (ms)": s["jev_avg_latency_ms"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _results(job: dict[str, Any]) -> None:
    results = job.get("results") or {}
    if job.get("error"):
        st.error(job["error"])
    if not results:
        return
    for c, r in results.items():
        if r.get("error"):
            st.warning(f"{CONTROLLER_LABELS.get(c, c)} failed: {r['error']}")
    _headline(results)
    st.plotly_chart(charts.benchmark_bars(results), use_container_width=True)
    st.plotly_chart(charts.approach_delay_bars(results), use_container_width=True)
    st.markdown("#### Full comparison")
    _table(results)


def render() -> None:
    info = require_api()
    header(
        "Benchmark",
        "Every controller faces the exact same vehicles, drivers and pedestrians, and is graded on ground truth.",
    )
    _form(info)
    job_id = st.session_state.get(JOB_KEY)
    if job_id:
        try:
            _results(_wait(job_id))
        except ApiError as exc:
            st.error(str(exc))
    past = [b for b in api().benchmarks() if b["status"] == "finished"]
    if past:
        st.divider()
        st.markdown("#### Previous benchmarks")
        labels = {b["id"]: f"{b['created_at']} · {b['scenario']} · seed {b['seed']}" for b in past}
        pick = st.selectbox(
            "Open a previous benchmark", [None, *labels], format_func=lambda k: labels.get(k, "—")
        )
        if pick:
            _results(api().benchmark(pick))
