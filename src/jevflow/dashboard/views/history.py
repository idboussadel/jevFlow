"""Run history: every live and benchmark run, with its metrics and full decision audit trail."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from jevflow.dashboard import charts
from jevflow.dashboard.state import api, header, require_api
from jevflow.dashboard.theme import CONTROLLER_LABELS


def render() -> None:
    require_api()
    header(
        "Run history",
        "Everything is stored in SQLite: one-second metrics and every decision, including Jev's input.",
    )
    runs = api().runs(limit=200)
    if not runs:
        st.info("No runs yet. Start one from the live control room or the benchmark page.")
        return
    table = pd.DataFrame(
        [
            {
                "id": r["id"],
                "created": r["created_at"],
                "scenario": r["scenario"],
                "controller": CONTROLLER_LABELS.get(r["controller"], r["controller"]),
                "mode": r["mode"],
                "status": r["status"],
                "avg delay (s)": (r["summary"] or {}).get("avg_control_delay_s"),
                "LOS": (r["summary"] or {}).get("level_of_service"),
                "served": (r["summary"] or {}).get("vehicles_served"),
            }
            for r in runs
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True, height=280)
    run_id = st.selectbox("Inspect run", table["id"].tolist(), format_func=lambda i: f"{i} · " + " · ".join(
        str(v) for v in table.loc[table["id"] == i, ["scenario", "controller", "created"]].iloc[0]
    ))  # fmt: skip
    run = next(r for r in runs if r["id"] == run_id)
    summary = run.get("summary") or {}
    if summary:
        cols = st.columns(5)
        cols[0].metric(
            "Avg control delay",
            f"{summary['avg_control_delay_s']} s",
            f"LOS {summary['level_of_service']}",
            delta_color="off",
        )
        cols[1].metric("95th pct delay", f"{summary['p95_delay_s']} s")
        cols[2].metric("Throughput", f"{summary['throughput_vph']:.0f} veh/h")
        cols[3].metric("Stops / vehicle", summary["stops_per_vehicle"])
        cols[4].metric("Queue estimate MAE", f"{summary['queue_estimate_mae']} veh")
    samples = api().run_metrics(run_id)
    if samples:
        st.markdown("#### Queue per approach · detector estimate vs ground truth")
        st.plotly_chart(charts.queue_small_multiples(samples), use_container_width=True)
        st.plotly_chart(charts.delay_trend(samples), use_container_width=True)
    decisions = api().run_decisions(run_id, limit=500)
    if decisions:
        st.markdown("#### Decisions")
        jev_only = st.toggle("Only decisions that called Jev", value=any("jev" in d for d in decisions))
        shown = [d for d in decisions if not jev_only or "jev" in d]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "t (s)": d["t"],
                        "decision": d["action"],
                        "applied": d["applied"],
                        "source": d["source"],
                        "confidence": d["confidence"],
                        "latency (ms)": d["latency_ms"],
                        "why": d["rationale"],
                    }
                    for d in shown
                ]
            ),
            hide_index=True,
            use_container_width=True,
            height=300,
            column_config={
                "confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")
            },
        )
        with_jev = [d for d in shown if d.get("jev")]
        if with_jev:
            pick = st.select_slider(
                "Inspect a Jev call", options=list(range(len(with_jev))),
                format_func=lambda i: f"t={with_jev[i]['t']}s → {with_jev[i]['action']}",
            )  # fmt: skip
            d = with_jev[pick]
            left, right = st.columns(2)
            left.markdown("**State sent to Jev**")
            left.json(d["jev"]["state"], expanded=False)
            right.markdown("**Questions and answer**")
            right.json(
                {"questions": d["jev"]["questions"], "probabilities": d["probabilities"], "confidence": d["confidence"],
                 "congestion_score": d["jev"]["congestion_score"], "error": d["jev"]["error"]},
                expanded=False,
            )  # fmt: skip
