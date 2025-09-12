# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- Your helpers (provide in services.py) -----------------------------------
# Expected signatures:
#   get_news_data(type_of_search: str, search_string: str, date_preset: str) -> dict
#   generate_response(llm_client_or_token, text: str, system_prompt: str) -> dict  # {"content": "..."}
#   extract_sentiment_score(text: str) -> Optional[float]
#   get_oauth_token() -> str | client (whatever your generate_response expects)
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
def init_state() -> None:
    st.session_state.setdefault("mode", "Company")
    st.session_state.setdefault("company_name", COMPANIES[0]["name"])
    st.session_state.setdefault("topic_name", "")
    st.session_state.setdefault("date_preset", "this_week")

    st.session_state.setdefault("polling", False)          # drives autorefresh
    st.session_state.setdefault("news", None)              # {"docs": df, "info": {...}}
    st.session_state.setdefault("_docs_sig", "")           # fingerprint of current docs
    st.session_state.setdefault("summary_stale", True)     # regenerate summary when True
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)
    st.session_state.setdefault("last_update", None)

    # simple JSONL logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    st.session_state.setdefault("_log_file", log_dir / f"polling-{datetime.today():%Y%m%d}.log")

def log_event(event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    try:
        file: Path = st.session_state["_log_file"]
        data = {"ts": datetime.utcnow().isoformat(), "event": event, "payload": payload or {}}
        with file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never break the UI for logging

# --- Utilities ---------------------------------------------------------------
def colour_for_score(score: float) -> str:
    score = max(-1.0, min(1.0, float(score)))
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = mcolors.get_cmap("RdYlGn")
    return mcolors.to_hex(cmap(norm(score)))

def news_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty:0"
    cols = [c for c in ("id", "timestamp", "url", "title") if c in df.columns]
    if not cols:
        return f"rows:{len(df)}"
    return str(int(pd.util.hash_pandas_object(df[cols].astype(str), index=False).sum()))

def fetch_and_update_news() -> None:
    """Fetch, normalise, update state; mark summary stale if content changed."""
    mode = st.session_state.mode
    preset_key = st.session_state.date_preset
    preset_val = DATE_PRESETS[preset_key]

    if mode == "Company":
        subject = (st.session_state.company_name or "").strip()
        if not subject:
            st.warning("Please select a company.")
            return
        res = get_news_data("COMPANY", subject, preset_val)
        subtitle = f"Company – {subject}"
    else:
        subject = (st.session_state.topic_name or "").strip()
        if not subject:
            st.warning("Please enter a topic.")
            return
        res = get_news_data("TOPIC", subject, preset_val)
        subtitle = f"Topic – {subject}"

    df = res.get("res") if isinstance(res, dict) else pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    # Normalise dtypes
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)

    # Detect changes
    new_sig = news_signature(df)
    changed = new_sig != st.session_state["_docs_sig"]
    if changed:
        st.session_state["_docs_sig"] = new_sig
        st.session_state["summary_stale"] = True

    st.session_state["news"] = {"docs": df, "info": {"subtitle": subtitle}}
    st.session_state["last_update"] = datetime.utcnow()
    log_event("news_fetched", {"rows": int(len(df)), "subject": subject, "changed": changed})

def maybe_poll() -> None:
    """Called on every app run. When polling is ON, fetch & flag changes."""
    if not st.session_state.polling:
        return
    prev_sig = st.session_state["_docs_sig"]
    fetch_and_update_news()
    new_sig = st.session_state["_docs_sig"]
    if new_sig != prev_sig:
        st.toast("New articles detected")
        log_event("poll_new_content", {"prev_sig": prev_sig, "new_sig": new_sig})
    else:
        log_event("poll_no_change", {"sig": new_sig})

def sentiment_scale(avg: float) -> go.Figure:
    avg = max(-1.0, min(1.0, float(avg)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1, 1], y=[0, 0], mode="lines",
                             line=dict(color="gray", width=2), name="Scale"))
    fig.add_trace(go.Scatter(x=[avg], y=[0], mode="markers",
                             marker=dict(color=colour_for_score(avg), size=14),
                             name="Avg"))
    fig.update_layout(
        title="Sentiment Score",
        xaxis=dict(range=[-1, 1], tickvals=[-1, 0, 1], ticktext=["−1", "0", "+1"]),
        yaxis=dict(visible=False), showlegend=False, height=150, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# --- App ---------------------------------------------------------------------
def main() -> None:
    init_state()

    # 🔁 Polling toggle and status
    top1, top2, top3 = st.columns([1, 1, 1])
    last = st.session_state.last_update
    top1.caption(f"Last refresh: {last:%H:%M:%S} UTC" if last else "Last refresh: —")
    top2.caption("Polling: ✅" if st.session_state.polling else "Polling: ⛔")
    if top3.button("🔁 Toggle Polling", use_container_width=True):
        st.session_state.polling = not st.session_state.polling
        st.toast("Polling started" if st.session_state.polling else "Polling stopped")
        log_event("poll_toggle", {"on": st.session_state.polling})

    # ✅ Auto-refresh only when polling is ON
    if st.session_state.polling:
        # Returns an incrementing counter we don't actually need; the side-effect is the rerun.
        st_autorefresh(interval=POLL_INTERVAL * 1000, key="poller")

    st.title("Sentiment Analysis")
    st.divider()

    # --- Sidebar (state-persistent widgets) ---------------------------------
    with st.sidebar:
        st.header("Scope")

        # Mode
        st.session_state.mode = st.radio(
            "Analyse by", ["Company", "Topic"],
            index=0 if st.session_state.mode == "Company" else 1
        )

        # Date preset
        preset_keys = list(DATE_PRESETS.keys())
        current_preset = st.session_state.date_preset
        st.session_state.date_preset = st.selectbox(
            "Date range (UTC)",
            preset_keys,
            index=preset_keys.index(current_preset) if current_preset in preset_keys else 0,
        )

        # Subject
        if st.session_state.mode == "Company":
            company_names = [c["name"] for c in COMPANIES]
            current_company = st.session_state.company_name or company_names[0]
            st.session_state.company_name = st.selectbox(
                "Company",
                company_names,
                index=company_names.index(current_company) if current_company in company_names else 0,
            )
        else:
            st.session_state.topic_name = st.text_input(
                "Topic", value=st.session_state.topic_name
            )

        # Manual fetch button
        if st.button("🧠 Get Sentiment Analysis", use_container_width=True):
            fetch_and_update_news()

    # --- Automatic poll on each run (only does work when polling ON) --------
    maybe_poll()

    # --- Render results ------------------------------------------------------
    news = st.session_state.news or {}
    df = news.get("docs", pd.DataFrame())
    subtitle = news.get("info", {}).get("subtitle", "")

    if not df.empty:
        st.subheader(subtitle)

        # KPIs
        if "sentiment" in df.columns:
            s = pd.to_numeric(df["sentiment"], errors="coerce")
            avg = float(s.mean()) if not s.empty else 0.0
            std = float(s.std()) if not s.empty else 0.0
        else:
            avg = std = 0.0

        k1, k2 = st.columns(2)
        k1.metric("Avg Sentiment", f"{avg:.2f}")
        k2.metric("Std Dev", f"{std:.2f}")
        st.plotly_chart(sentiment_scale(avg), use_container_width=True)

        # Charts
        if "sentiment" in df.columns:
            st.plotly_chart(px.histogram(df, x="sentiment", nbins=30, title="Sentiment Distribution"),
                            use_container_width=True)

        if "timestamp" in df.columns and "sentiment" in df.columns:
            ts = df.sort_values("timestamp").copy()
            ts["ma"] = pd.to_numeric(ts["sentiment"], errors="coerce").expanding().mean()
            fig = px.scatter(ts, x="timestamp", y="sentiment", title="Sentiment Over Time")
            fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["ma"], mode="lines", name="Moving Avg"))
            fig.add_hline(y=0, line_color="gray", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

        # Table
        cfg = {}
        if "url" in df.columns:
            cfg["url"] = st.column_config.LinkColumn(display_text="Link")
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=cfg)
    else:
        st.info("No news yet. Choose a scope and click **Get Sentiment Analysis**.")

    # --- Executive summary ---------------------------------------------------
    if not df.empty:
        st.subheader("Executive Summary")
        if st.session_state.summary_stale:
            chunks = df.to_json(orient="records")
            prompt = (
                "You are a content summarising assistant.\n"
                "Create a concise executive summary of the following items and end with a line:\n"
                "**Sentiment Score:** <NUMBER between -1 and 1>\n"
            )
            client_or_token = get_oauth_token()  # adapt to your generate_response
            with st.spinner("Summarising..."):
                resp = generate_response(client_or_token, chunks, prompt)
                content = resp.get("content", "")
                score = extract_sentiment_score(content)
                st.session_state.summary_text = content
                st.session_state.llm_sentiment = float(score) if score is not None else None
                st.session_state.summary_stale = False
            log_event("summary_generated", {"sig": st.session_state["_docs_sig"]})

        if st.session_state.llm_sentiment is not None:
            st.metric("LLM Sentiment", f"{st.session_state.llm_sentiment:.2f}")
        st.markdown(st.session_state.summary_text)

    # --- Footer --------------------------------------------------------------
    st.caption("© MAAS Execution Analytics")

if __name__ == "__main__":
    main()
