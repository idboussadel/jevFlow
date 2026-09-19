"""Streamlit wrapper for the live intersection canvas (a self-contained HTML/JS component)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import streamlit.components.v1 as components

_TEMPLATE = Path(__file__).with_name("static") / "intersection.html"


@cache
def _template() -> str:
    return _TEMPLATE.read_text(encoding="utf-8")


def live_intersection(session_id: str, ws_url: str, height: int = 780) -> None:
    """Render the live view. The browser connects to the API WebSocket directly, so the
    animation runs at display refresh rate independently of Streamlit reruns."""
    html = (
        _template()
        .replace('"__WS_URL__"', json.dumps(ws_url.rstrip("/")))
        .replace('"__SESSION_ID__"', json.dumps(session_id))
    )
    components.html(html, height=height, scrolling=False)
