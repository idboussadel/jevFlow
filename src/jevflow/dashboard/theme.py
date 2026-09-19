"""Visual system for the dashboard: tokens, Plotly template, and page CSS.

The four controllers use categorical slots 1–4 of the validated reference palette
(dark steps, checked against the ``#0f1115`` surface with the palette validator). A
controller keeps its colour on every chart, so readers learn it once.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#0f1115"
SURFACE_2 = "#151923"
BORDER = "rgba(255,255,255,0.07)"
TEXT = "#e8e9ee"
TEXT_2 = "#b4b6c0"
MUTED = "#7c7f8c"
GRID = "rgba(255,255,255,0.06)"
TRUTH = "#d6d7dd"  # neutral ink for ground truth, never a series hue

CONTROLLER_COLORS: dict[str, str] = {
    "jev": "#3987e5",
    "actuated": "#d95926",
    "max_pressure": "#199e70",
    "fixed_time": "#c98500",
}
CONTROLLER_LABELS: dict[str, str] = {
    "jev": "Jev",
    "actuated": "Actuated",
    "max_pressure": "Max-pressure",
    "fixed_time": "Fixed-time",
}
ESTIMATE = CONTROLLER_COLORS["jev"]

# Status tokens: reserved for meaning (good → critical), always shown with a label.
STATUS = {"good": "#2fb67c", "warning": "#e0a526", "serious": "#e5733a", "critical": "#e5484d"}
LOS_STATUS = {"A": "good", "B": "good", "C": "warning", "D": "warning", "E": "serious", "F": "critical"}


def _register_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": "Inter, system-ui, sans-serif", "color": TEXT_2, "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(CONTROLLER_COLORS.values()),
        margin={"l": 48, "r": 16, "t": 72, "b": 36},
        hoverlabel={"bgcolor": SURFACE_2, "bordercolor": BORDER, "font": {"color": TEXT, "size": 12}},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
            "x": 0,
            "font": {"color": TEXT_2},
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={"gridcolor": GRID, "zeroline": False, "linecolor": BORDER, "tickcolor": BORDER, "ticks": ""},
        yaxis={"gridcolor": GRID, "zeroline": False, "linecolor": BORDER, "ticks": "", "rangemode": "tozero"},
        title={"font": {"color": TEXT, "size": 14}, "x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top"},
    )
    pio.templates["jevflow"] = template
    pio.templates.default = "jevflow"


_register_template()

PAGE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] {{ font-family: Inter, system-ui, sans-serif; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1480px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
[data-testid="stMetric"] {{
  background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }}
[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; }}
.jf-hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:24px; margin-bottom: 6px; }}
.jf-title {{ font-size: 2.1rem; font-weight: 700; margin: 0; letter-spacing: -0.03em; line-height: 1.15; color: {TEXT}; }}
.jf-title span {{ color: {ESTIMATE}; }}
.jf-sub {{ color: {TEXT_2}; margin: 6px 0 0; font-size: 0.98rem; }}
.jf-chip {{ display:inline-flex; align-items:center; gap:6px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid {BORDER}; background: {SURFACE_2}; color: {TEXT_2}; font-size: .78rem; }}
.jf-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.jf-card {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:16px; padding:18px 20px; }}
.jf-card h4 {{ margin: 0 0 6px; font-size: 1rem; }}
.jf-card p {{ color:{TEXT_2}; margin:0; font-size:.9rem; line-height:1.5; }}
.jf-pipeline {{ display:grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 18px 0 8px; }}
.jf-step {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:14px; padding:14px; position:relative; }}
.jf-step b {{ display:block; color:{TEXT}; font-size:.92rem; margin-bottom:4px; }}
.jf-step small {{ color:{MUTED}; font-size:.8rem; line-height:1.45; display:block; }}
.jf-step .n {{ font-family:'JetBrains Mono', monospace; color:{ESTIMATE}; font-size:.75rem; }}
.jf-big {{ font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; font-family: 'JetBrains Mono', monospace; }}
.jf-muted {{ color:{MUTED}; }}
@media (max-width: 900px) {{ .jf-pipeline {{ grid-template-columns: 1fr 1fr; }} }}
</style>
"""
