"""Streamlit entry point: ``streamlit run src/jevflow/dashboard/app.py`` (or ``jevflow dashboard``)."""

from __future__ import annotations

import streamlit as st

from jevflow.dashboard.theme import PAGE_CSS
from jevflow.dashboard.views import about, benchmark, history, live

st.set_page_config(page_title="JevFlow · AI traffic signals", page_icon="🚦", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

navigation = st.navigation(
    [
        st.Page(live.render, title="Live control room", icon=":material/traffic:", default=True),
        st.Page(benchmark.render, title="Benchmark", icon=":material/leaderboard:", url_path="benchmark"),
        st.Page(history.render, title="Run history", icon=":material/history:", url_path="history"),
        st.Page(about.render, title="How it works", icon=":material/schema:", url_path="about"),
    ]
)
navigation.run()
