# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st

# --- External helpers (in your services.py) ---------------------------------
from services import (
    get_oauth_token,
    get_news_data,
    generate_response,
    extract_sentiment_score,
)

# --- Page config -------------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

# --- Constants ---------------------------------------------------------------
COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
]

date_preset_dict = {
    "today": "TODAY",
    "yesterday": "YESTERDAY",
    "last_24_hours": "LAST_TWENTY_FOUR_HOURS",
    "this_week": "THIS_WEEK",
    "last_week": "LAST_WEEK",
    "last_thirty_days": "LAST_THIRTY_DAYS",
}

POLL_INTERVAL = 60  # seconds


# --- Session state init ------------------------------------------------------
def init_state():
    st.session_state.setdefault("mode", "Company")
    st.session_state.setdefault("company_name", "")
    st.session_state.setdefault("topic_name", "")
    st.session_state.setdefault("date_preset", "this_week")
    st.session_state.setdefault("polling", False)
    st.session_state.setdefault("next_poll_at", None)

    st.session_state.setdefault("news", None)
    st.session_state.setdefault("_docs_sig", "")
    st.session_state.setdefault("summary_stale", True)

    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)
    st.session_state.setdefault("last_update", None)


# --- Helpers -----------------------------------------------------------------
def colour_for_score(score: float) -> str:
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = mcolors.get_cmap("RdYlGn")
    rgba = cmap(norm(score))
    return mcolors.to_hex(rgba)


def news_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty:0"
    cols = [c for c in ("id", "timestamp", "url", "title") if c in df.columns]
    if not cols:
        return f"rows:{len(df)}"
    return str(int(pd.util.hash_pandas_object(df[cols].astype(str), index=False).sum()))


def fetch_and_update_news():
    """Fetch data, normalise, and mark summary stale if content changed."""
    mode = st.session_state.mode
    preset = st.session_state.date_preset

    if mode == "Company":
        subject = st.session_state.company_name.strip()
        if not subject:
            st.warning("Please select a company")
            return
        res = get_news_data("COMPANY", subject, date_preset_dict[preset])
        subtitle = f"Company – {subject}"
    else:
        subject = st.session_state.topic_name.strip()
        if not subject:
            st.warning("Please enter a topic")
            return
        res = get_news_data("TOPIC", subject, date_preset_dict[preset])
        subtitle = f"Topic – {subject}"

    df = res.get("res") if isinstance(res, dict) else pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    new_sig = news_signature(df)
    if new_sig != st.session_state["_docs_sig"]:
        st.session_state["_docs_sig"] = new_sig
        st.session_state["summary_stale"] = True

    st.session_state["news"] = {"docs": df, "info": {"subtitle": subtitle}}
    st.session_state["last_update"] = datetime.utcnow()


def maybe_poll():
    """Run on each script run; fetch new news if polling and due."""
    if not st.session_state.polling:
        return
    now = datetime.utcnow()
    if st.session_state.next_poll_at is None or now >= st.session_state.next_poll_at:
        prev_sig = st.session_state["_docs_sig"]
        fetch_and_update_news()
        st.session_state.next_poll_at = now + timedelta(seconds=POLL_INTERVAL)
        if st.session_state["_docs_sig"] != prev_sig:
            st.toast("New articles detected — refreshing…")
            st.experimental_rerun()


def sentiment_scale(avg: float) -> go.Figure:
    avg = max(-1.0, min(1.0, float(avg)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1, 1], y=[0, 0], mode="lines",
                             line=dict(color="gray", width=2)))
    fig.add_trace(go.Scatter(x=[avg], y=[0], mode="markers",
                             marker=dict(color=colour_for_score(avg), size=15)))
    fig.update_layout(
        title="Sentiment Score",
        xaxis=dict(range=[-1, 1],
                   tickvals=[-1, 0, 1],
                   ticktext=["−1", "0", "+1"]),
        yaxis=dict(visible=False),
        showlegend=False,
        height=160,
    )
    return fig


# --- Main --------------------------------------------------------------------
def main():
    init_state()
    maybe_poll()

    st.title("Sentiment Analysis")

    # Top status row
    c1, c2, c3 = st.columns([1, 1, 1])
    last = st.session_state.last_update
    c1.caption(f"Last refresh: {last:%H:%M:%S}" if last else "Last refresh: —")
    if st.session_state.polling:
        c2.caption("Polling: ✅")
    else:
        c2.caption("Polling: ⛔")
    if c3.button("🔁 Toggle Polling"):
        st.session_state.polling = not st.session_state.polling
        if st.session_state.polling:
            st.session_state.next_poll_at = datetime.utcnow()
            st.toast("Polling started")
        else:
            st.toast("Polling stopped")
        st.experimental_rerun()

    st.divider()

    # --- Sidebar controls
    with st.sidebar:
    st.header("Scope")

    # Mode selection
    st.session_state.mode = st.radio(
        "Analyse by",
        ["Company", "Topic"],
        index=0 if st.session_state.mode == "Company" else 1,
    )

    # Date preset selection
    preset_keys = list(date_preset_dict.keys())
    current_preset = st.session_state.get("date_preset", preset_keys[0])
    st.session_state.date_preset = st.selectbox(
        "Date range",
        preset_keys,
        index=preset_keys.index(current_preset) if current_preset in preset_keys else 0,
    )

    # Company or topic input
    if st.session_state.mode == "Company":
        company_names = [c["name"] for c in COMPANIES]
        current_company = st.session_state.get("company_name") or company_names[0]
        st.session_state.company_name = st.selectbox(
            "Company",
            company_names,
            index=company_names.index(current_company)
            if current_company in company_names
            else 0,
        )
    else:
        st.session_state.topic_name = st.text_input(
            "Topic",
            value=st.session_state.get("topic_name", ""),
        )

    if st.button("🧠 Get Sentiment Analysis"):
        fetch_and_update_news()

    # --- Data + charts
    news = st.session_state.news or {}
    df = news.get("docs", pd.DataFrame())
    subtitle = news.get("info", {}).get("subtitle", "")

    if not df.empty:
        st.subheader(subtitle)

        # KPIs
        avg = float(df["sentiment"].mean()) if "sentiment" in df else 0.0
        std = float(df["sentiment"].std()) if "sentiment" in df else 0.0
        c1, c2 = st.columns(2)
        c1.metric("Avg Sentiment", f"{avg:.2f}")
        c2.metric("Std Dev", f"{std:.2f}")
        st.plotly_chart(sentiment_scale(avg), use_container_width=True)

        # Charts
        if "sentiment" in df.columns:
            st.plotly_chart(px.histogram(df, x="sentiment", nbins=30), use_container_width=True)
        if "timestamp" in df.columns and "sentiment" in df.columns:
            ts = df.sort_values("timestamp")
            ts["ma"] = ts["sentiment"].expanding().mean()
            fig = px.scatter(ts, x="timestamp", y="sentiment")
            fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["ma"], mode="lines", name="MA"))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No news yet. Select scope and click **Get Sentiment Analysis**.")

    # --- Executive summary
    if not df.empty:
        st.subheader("Executive Summary")
        if st.session_state.summary_stale:
            chunks = df.to_json(orient="records")
            prompt = (
                "Summarise the following news.\n"
                "End with one line: **Sentiment Score:** <NUMBER between -1 and 1>\n"
            )
            client = get_oauth_token()  # example; adapt for your client
            with st.spinner("Summarising..."):
                resp = generate_response(client, chunks, prompt)
                content = resp.get("content", "")
                score = extract_sentiment_score(content)
                st.session_state.summary_text = content
                st.session_state.llm_sentiment = score
                st.session_state.summary_stale = False

        if st.session_state.llm_sentiment is not None:
            st.metric("LLM Sentiment", f"{st.session_state.llm_sentiment:.2f}")
        st.markdown(st.session_state.summary_text)


if __name__ == "__main__":
    main()
