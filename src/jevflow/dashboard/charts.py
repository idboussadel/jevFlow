"""Plotly figure builders. One job per chart, one y-axis per chart, recessive chrome."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jevflow.dashboard.theme import (
    CONTROLLER_COLORS,
    CONTROLLER_LABELS,
    ESTIMATE,
    GRID,
    MUTED,
    TEXT,
    TEXT_2,
    TRUTH,
)

APPROACHES = ("north", "south", "east", "west")
LOS_BANDS = ((10, "A"), (20, "B"), (35, "C"), (55, "D"), (80, "E"))


def _minutes(samples: list[dict[str, Any]]) -> list[float]:
    return [s["t"] / 60.0 for s in samples]


def queue_small_multiples(samples: list[dict[str, Any]], height: int = 300) -> go.Figure:
    """Detector estimate vs ground truth per approach: the honesty chart."""
    fig = make_subplots(
        rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
        subplot_titles=[a.capitalize() for a in APPROACHES],
    )  # fmt: skip
    x = _minutes(samples)
    for col, approach in enumerate(APPROACHES, start=1):
        truth = [s["true_queue"][approach] for s in samples]
        estimate = [s["estimated_queue"][approach] for s in samples]
        fig.add_trace(
            go.Scatter(
                x=x, y=truth, name="Ground truth", mode="lines", legendgroup="truth", showlegend=col == 1,
                line={"color": TRUTH, "width": 1.6, "dash": "dot"},
                hovertemplate="truth %{y:.0f} veh<extra></extra>",
            ),
            row=1, col=col,
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=x, y=estimate, name="Detector estimate (what the controller sees)", mode="lines",
                legendgroup="est", showlegend=col == 1, line={"color": ESTIMATE, "width": 2},
                hovertemplate="estimate %{y:.1f} veh<extra></extra>",
            ),
            row=1, col=col,
        )  # fmt: skip
    fig.update_layout(
        height=height,
        margin={"l": 40, "r": 8, "t": 72, "b": 32},
        hovermode="x unified",
        legend={"y": 1.16, "yanchor": "bottom"},
    )
    fig.update_xaxes(title_text="", ticksuffix=" min", gridcolor=GRID)
    fig.update_yaxes(title_text="queued vehicles", row=1, col=1)
    for annotation in fig.layout.annotations:
        annotation.font = {"color": TEXT_2, "size": 12}
    return fig


def delay_trend(samples: list[dict[str, Any]], height: int = 300) -> go.Figure:
    fig = go.Figure()
    x = _minutes(samples)
    y = [s["avg_delay_recent_s"] for s in samples]
    top = max([*y, 60.0]) * 1.1
    for limit, grade in LOS_BANDS:
        if limit < top:
            fig.add_hline(y=limit, line={"color": GRID, "width": 1})
            fig.add_annotation(
                x=1.0, xref="paper", y=limit, text=f"LOS {grade}", showarrow=False, xanchor="right",
                yanchor="bottom", font={"color": MUTED, "size": 10},
            )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="Control delay", line={"color": ESTIMATE, "width": 2},
            fill="tozeroy", fillcolor="rgba(57,135,229,0.10)",
            hovertemplate="%{y:.1f} s/veh<extra></extra>",
        )
    )  # fmt: skip
    fig.update_layout(
        height=height,
        showlegend=False,
        title={"text": "Average control delay · vehicles cleared in the last 60 s"},
        yaxis={"title": "seconds per vehicle", "range": [0, top]},
        xaxis={"ticksuffix": " min"},
    )
    return fig


def throughput_trend(samples: list[dict[str, Any]], height: int = 300) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=_minutes(samples), y=[s["throughput_vph"] for s in samples], mode="lines",
            line={"color": ESTIMATE, "width": 2}, hovertemplate="%{y:.0f} veh/h<extra></extra>",
        )
    )  # fmt: skip
    fig.update_layout(
        height=height, showlegend=False,
        title={"text": "Throughput · rolling 5 min"},
        yaxis={"title": "vehicles per hour"}, xaxis={"ticksuffix": " min"},
    )  # fmt: skip
    return fig


METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("avg_control_delay_s", "Avg control delay", "s/veh", False),
    ("p95_delay_s", "95th-percentile delay", "s", False),
    ("max_wait_s", "Longest single wait", "s", False),
    ("stops_per_vehicle", "Stops per vehicle", "", False),
    ("throughput_vph", "Throughput", "veh/h", True),
    ("fairness_index", "Fairness (Jain)", "", True),
)


def benchmark_bars(results: dict[str, dict[str, Any]], height: int = 520) -> go.Figure:
    """Small multiples: one metric per panel, controllers keep their colour everywhere."""
    controllers = [c for c, r in results.items() if r.get("summary")]
    fig = make_subplots(
        rows=2, cols=3, vertical_spacing=0.2, horizontal_spacing=0.07,
        subplot_titles=[
            f"{label}{' (' + unit + ')' if unit else ''} · {'higher' if hib else 'lower'} is better"
            for _, label, unit, hib in METRICS
        ],
    )  # fmt: skip
    for i, (key, label, unit, _) in enumerate(METRICS):
        row, col = divmod(i, 3)
        values = [results[c]["summary"][key] for c in controllers]
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_LABELS.get(c, c) for c in controllers],
                y=values,
                marker={"color": [CONTROLLER_COLORS.get(c, MUTED) for c in controllers], "cornerradius": 4},
                text=[_fmt(v) for v in values],
                textposition="outside",
                textfont={"color": TEXT, "size": 11},
                cliponaxis=False,
                hovertemplate=f"%{{x}}: %{{y}} {unit}<extra>{label}</extra>",
                showlegend=False,
            ),
            row=row + 1, col=col + 1,
        )  # fmt: skip
    fig.update_layout(
        height=height, bargap=0.35, hovermode="closest", margin={"l": 32, "r": 12, "t": 48, "b": 24}
    )
    fig.update_yaxes(showticklabels=False, gridcolor="rgba(0,0,0,0)")
    for annotation in fig.layout.annotations:
        annotation.font = {"color": TEXT_2, "size": 12}
    return fig


def approach_delay_bars(results: dict[str, dict[str, Any]], height: int = 320) -> go.Figure:
    fig = go.Figure()
    for controller, result in results.items():
        summary = result.get("summary")
        if not summary:
            continue
        fig.add_trace(
            go.Bar(
                name=CONTROLLER_LABELS.get(controller, controller),
                x=[a.capitalize() for a in APPROACHES],
                y=[summary["approach_delay_s"][a] for a in APPROACHES],
                marker={"color": CONTROLLER_COLORS.get(controller, MUTED), "cornerradius": 4},
                hovertemplate="%{x}: %{y:.1f} s<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        height=height, barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest",
        title={"text": "Average delay by approach · who pays for the decisions"},
        yaxis={"title": "seconds per vehicle"},
    )  # fmt: skip
    return fig


def _fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"
