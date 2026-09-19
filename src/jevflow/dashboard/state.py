"""Shared dashboard helpers: API client, cached lookups, and small UI building blocks."""

from __future__ import annotations

from typing import Any

import streamlit as st

from jevflow.config import get_settings
from jevflow.dashboard.api_client import ApiError, JevFlowApi
from jevflow.dashboard.theme import CONTROLLER_COLORS, CONTROLLER_LABELS, LOS_STATUS, STATUS


@st.cache_resource
def api() -> JevFlowApi:
    return JevFlowApi(get_settings().api_url)


@st.cache_data(ttl=30, show_spinner=False)
def scenarios() -> list[dict[str, Any]]:
    return api().scenarios()


@st.cache_data(ttl=10, show_spinner=False)
def controllers() -> list[dict[str, Any]]:
    return api().controllers()


@st.cache_data(ttl=10, show_spinner=False)
def health() -> dict[str, Any] | None:
    try:
        return api().health()
    except ApiError:
        return None


def require_api() -> dict[str, Any]:
    info = health()
    if info is None:
        st.error(
            f"The JevFlow API is not reachable at `{get_settings().api_url}`. "
            "Start it with `uv run jevflow api` (or `uv run jevflow up` for API + dashboard).",
            icon=":material/cloud_off:",
        )
        st.stop()
    return info


def controller_chip(key: str) -> str:
    color = CONTROLLER_COLORS.get(key, "#7c7f8c")
    label = CONTROLLER_LABELS.get(key, key)
    return f'<span class="jf-chip"><span class="jf-dot" style="background:{color}"></span>{label}</span>'


def los_chip(grade: str) -> str:
    color = STATUS[LOS_STATUS.get(grade, "critical")]
    return f'<span class="jf-chip" style="color:{color};border-color:{color}55"><b>LOS {grade}</b></span>'


def header(title_html: str, subtitle: str, right_html: str = "") -> None:
    st.markdown(
        f'<div class="jf-hero"><div><div class="jf-title">{title_html}</div><div class="jf-sub">{subtitle}</div></div>'
        f"<div>{right_html}</div></div>",
        unsafe_allow_html=True,
    )
