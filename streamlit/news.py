# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- External helpers (implement in services.py) -----------------------------
from services import (
    get_oauth_token,
    get_news_data,
    generate_response,
    extract_sentiment_score,
)

# --- Config -----------------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
]

DATE_PRESETS = {
    "today": "TODAY",
    "yesterday": "YESTERDAY",
    "last_24_hours": "LAST_TWENTY_FOUR_HOURS",
    "this_week": "THIS_WEEK",
    "last_week": "LAST_WEEK",
    "last_thirty_days": "LAST_THIRTY_DAYS",
}

POLL_INTERVAL = 60  # seconds

# --- State & logging ---------------------------------------------------------
def init_state():
    st.session_state.setdefault("mode", "Company")
    st.session_state.setdefault("company_name", COMPANIES[0]["name"])
    st.session_state.setdefault("topic_name", "")
    st.session_state.setdefault("date_preset", "this_week")

    st.session_state.setdefault("polling", False)
    st.session_state.setdefault("news", None)

    # for new-content detection
    st.session_state.setdefault("_latest_ts", None)        # most recent timestamp seen
    st.session_state.setdefault("_latest_ts_count", 0)     # how many rows share that timestamp

    # summary state
    st.session_state.setdefault("summary_stale", True)
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)

    st.session_state.setdefault("last_update", None)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    st.session_state.setdefault("_log_file", log_dir / f"polling-{datetime.today():%Y%m%d}.log")


def log_event(event: str, payload: Optional[Dict[str, Any]] = None):
    try:
        file: Path = st.session_state["_log_file"]
        data = {"ts": datetime.utcnow().isoformat(), "event": event, "payload": payload or {}}
        with file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass

# --- Utilities ---------------------------------------------------------------
def colour_for_score(score: float) -> str:
    score = max(-1.0, min(1.0, float(score)))
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = mcolors.get_cmap("RdYlGn")
    return mcolors.to_hex(cmap(norm(score)))


def fetch_and_update_news():
    mode = st.session_state.mode
    preset_key = st.session_state.date_preset
    preset_val = DATE_PRESETS[preset_key]

    if mode == "Company":
        subject = st.session_state.company_name.strip()
        res = get_news_data("COMPANY", subject, preset_val)
        subtitle = f"Company – {subject}"
    else:
        subject = st.session_state.topic_name.strip()
        res = get_news_data("TOPIC", subject, preset_val)
        subtitle = f"Topic – {subject}"

    df = res.get("res") if isinstance(res, dict) else pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)

    # --- new content detection based on latest timestamp
    prev_latest = st.session_state.get("_latest_ts")
    prev_count = st.session_state.get("_latest_ts_count", 0)

    if not df.empty and "timestamp" in df.columns:
        latest_ts = df["timestamp"].max()
        latest_count = int((df["timestamp"] == latest_ts).sum())
    else:
        latest_ts = None
        latest_count = 0

    changed = False
    if latest_ts is not None:
        if (prev_latest is None) or (latest_ts > prev_latest):
            changed = True
        elif (latest_ts == prev_latest) and (latest_count > prev_count):
            changed = True

    # persist tip
    st.session_state["_latest_ts"] = latest_ts
    st.session_state["_latest_ts_count"] = latest_count

    if changed:
        st.session_state["summary_stale"] = True

    st.session_state["news"] = {"docs": df, "info": {"subtitle": subtitle}}
    st.session_state["last_update"] = datetime.utcnow()

    log_event("news_fetched", {
        "rows": int(len(df)),
        "subject": subject,
        "latest_ts": latest_ts.isoformat() if latest_ts is not None else None,
        "latest_count": latest_count,
        "changed": changed
    })


def maybe_poll():
    if not st.session_state.polling:
        return
    prev_tip = st.session_state.get("_latest_ts")
    fetch_and_update_news()
    if st.session_state.get("_latest_ts") != prev_tip:
        st.toast("Newer articles detected")
        log_event("poll_new_content", {"new_latest": str(st.session_state.get("_latest_ts"))})
    else:
        log_event("poll_no_change", {"latest": str(st.session_state.get("_latest_ts"))})


def sentiment_scale(avg: float) -> go.Figure:
    avg = max(-1.0, min(1.0, float(avg)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1, 1], y=[0, 0], mode="lines",
                             line=dict(color="gray", width=2)))
    fig.add_trace(go.Scatter(x=[avg], y=[0], mode="markers",
                             marker=dict(color=colour_for_score(avg), size=14)))
    fig.update_layout(
        title="Sentiment Score",
        xaxis=dict(range=[-1, 1], tickvals=[-1, 0, 1], ticktext=["−1", "0", "+1"]),
        yaxis=dict(visible=False), showlegend=False, height=150
    )
    return fig

# --- Main --------------------------------------------------------------------
def main():
    init_state()

    # --- Polling toggle
    c1, c2, c3 = st.columns([1, 1, 1])
    last = st.session_state.last_update
    c1.caption(f"Last refresh: {last:%H:%M:%S} UTC" if last else "Last refresh: —")
    c2.caption("Polling: ✅" if st.session_state.polling else "Polling: ⛔")
    if c3.button("🔁 Toggle Polling", use_container_width=True):
        st.session_state.polling = not st.session_state.polling
        st.toast("Polling started" if st.session_state.polling else "Polling stopped")
        log_event("poll_toggle", {"on": st.session_state.polling})
        st.experimental_rerun()

    # enable auto rerun only if polling ON
    if st.session_state.polling:
        st_autorefresh(interval=POLL_INTERVAL * 1000, key="poller")

    st.title("Sentiment Analysis")
    st.divider()

    # --- Sidebar controls
    with st.sidebar:
        st.header("Scope")
        st.session_state.mode = st.radio("Analyse by", ["Company", "Topic"],
                                         index=0 if st.session_state.mode == "Company" else 1)
        preset_keys = list(DATE_PRESETS.keys())
        st.session_state.date_preset = st.selectbox("Date range", preset_keys,
                                                    index=preset_keys.index(st.session_state.date_preset))
        if st.session_state.mode == "Company":
            company_names = [c["name"] for c in COMPANIES]
            st.session_state.company_name = st.selectbox("Company", company_names,
                                                         index=company_names.index(st.session_state.company_name))
        else:
            st.session_state.topic_name = st.text_input("Topic", st.session_state.topic_name)

        if st.button("🧠 Get Sentiment Analysis"):
            fetch_and_update_news()

    # poll check on every run
    maybe_poll()

    # --- Display results
    news = st.session_state.news or {}
    df = news.get("docs", pd.DataFrame())
    subtitle = news.get("info", {}).get("subtitle", "")

    if not df.empty:
        st.subheader(subtitle)

        avg = float(df["sentiment"].mean()) if "sentiment" in df else 0.0
        std = float(df["sentiment"].std()) if "sentiment" in df else 0.0

        k1, k2 = st.columns(2)
        k1.metric("Avg Sentiment", f"{avg:.2f}")
        k2.metric("Std Dev", f"{std:.2f}")
        st.plotly_chart(sentiment_scale(avg), use_container_width=True)

        if "sentiment" in df.columns:
            st.plotly_chart(px.histogram(df, x="sentiment", nbins=30), use_container_width=True)
        if "timestamp" in df.columns and "sentiment" in df.columns:
            ts = df.sort_values("timestamp").copy()
            ts["ma"] = pd.to_numeric(ts["sentiment"], errors="coerce").expanding().mean()
            fig = px.scatter(ts, x="timestamp", y="sentiment", title="Sentiment Over Time")
            fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["ma"], mode="lines", name="Moving Avg"))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No news yet. Choose a scope and click **Get Sentiment Analysis**.")

    # --- Executive summary
    if not df.empty:
        st.subheader("Executive Summary")
        if st.session_state.summary_stale:
            prompt = (
                "Summarise the following news items.\n"
                "End with one line: **Sentiment Score:** <NUMBER between -1 and 1>\n"
            )
            client = get_oauth_token()
            with st.spinner("Summarising..."):
                resp = generate_response(client, df.to_json(orient="records"), prompt)
                content = resp.get("content", "")
                score = extract_sentiment_score(content)
                st.session_state.summary_text = content
                st.session_state.llm_sentiment = score
                st.session_state.summary_stale = False
            log_event("summary_generated", {"latest_ts": str(st.session_state["_latest_ts"])})

        if st.session_state.llm_sentiment is not None:
            st.metric("LLM Sentiment", f"{st.session_state.llm_sentiment:.2f}")
        st.markdown(st.session_state.summary_text)

    st.caption("© MAAS Execution Analytics")


if __name__ == "__main__":
    main()
