# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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

# --- Config ------------------------------------------------------------------
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
def init_state() -> None:
    st.session_state.setdefault("mode", "Company")                 # "Company" | "Topic" | "Company + Topic"
    st.session_state.setdefault("company_input_mode", "Pick from list")  # "Pick from list" | "Type manually"

    st.session_state.setdefault("company_name", COMPANIES[2]["name"])  # default Alphabet
    st.session_state.setdefault("topic_name", "")
    st.session_state.setdefault("date_preset", "this_week")

    st.session_state.setdefault("polling", False)
    st.session_state.setdefault("news", None)

    # New-content detection (avoid false positives when rolling window drops old items)
    st.session_state.setdefault("_latest_ts", None)        # most recent timestamp seen (tz-aware)
    st.session_state.setdefault("_latest_ts_count", 0)     # how many rows share that timestamp

    # Summary state
    st.session_state.setdefault("summary_stale", True)     # mark True when new articles land
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)

    st.session_state.setdefault("last_update", None)

    # Simple JSONL logger
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
        pass  # never break UI due to logging

# --- Utilities ---------------------------------------------------------------
def colour_for_score(score: float) -> str:
    score = max(-1.0, min(1.0, float(score)))
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = mcolors.get_cmap("RdYlGn")
    return mcolors.to_hex(cmap(norm(score)))


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)
    return df


def _latest_tip(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], int]:
    if df.empty or "timestamp" not in df.columns:
        return None, 0
    latest_ts = df["timestamp"].max()
    latest_count = int((df["timestamp"] == latest_ts).sum())
    return latest_ts, latest_count


def fetch_and_update_news() -> None:
    """Fetch, normalise, and update state; mark summary stale only on truly newer content."""
    mode = st.session_state.mode
    preset_key = st.session_state.date_preset
    preset_val = DATE_PRESETS[preset_key]

    if mode == "Company":
        company = (st.session_state.company_name or "").strip()
        res = get_news_data("COMPANY", company, preset_val)
        subtitle = f"Company – {company}"

    elif mode == "Topic":
        topic = (st.session_state.topic_name or "").strip()
        res = get_news_data("TOPIC", topic, preset_val)
        subtitle = f"Topic – {topic}"

    elif mode == "Company + Topic":
        company = (st.session_state.company_name or "").strip()
        topic = (st.session_state.topic_name or "").strip()
        res = get_news_data("COMPANY_TOPIC", f"{company}|{topic}", preset_val)
        subtitle = f"Company + Topic – {company} / {topic}"

    else:
        res = {"res": pd.DataFrame(), "info": {}}
        subtitle = "Invalid mode"

    df = _normalise_df(res.get("res") if isinstance(res, dict) else pd.DataFrame())

    # Latest-tip change detection (prevents false positives from rolling windows)
    prev_latest = st.session_state.get("_latest_ts")
    prev_count = st.session_state.get("_latest_ts_count", 0)
    latest_ts, latest_count = _latest_tip(df)

    changed = False
    if latest_ts is not None:
        if (prev_latest is None) or (latest_ts > prev_latest):
            changed = True
        elif (latest_ts == prev_latest) and (latest_count > prev_count):
            changed = True

    st.session_state["_latest_ts"] = latest_ts
    st.session_state["_latest_ts_count"] = latest_count
    if changed:
        st.session_state["summary_stale"] = True

    st.session_state["news"] = {"docs": df, "info": {"subtitle": subtitle}}
    st.session_state["last_update"] = datetime.utcnow()

    log_event("news_fetched", {
        "rows": int(len(df)),
        "mode": mode,
        "subtitle": subtitle,
        "latest_ts": latest_ts.isoformat() if latest_ts is not None else None,
        "latest_count": latest_count,
        "changed": changed,
    })


def maybe_poll() -> None:
    """Called on every app run. When polling is ON, fetch & note changes."""
    if not st.session_state.polling:
        return
    before = st.session_state.get("_latest_ts")
    fetch_and_update_news()
    after = st.session_state.get("_latest_ts")
    if after and before and after > before:
        st.toast("Newer articles detected")
        log_event("poll_new_content", {"new_latest": after.isoformat()})
    else:
        log_event("poll_no_change", {"latest": after.isoformat() if after else None})


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

# --- Status panel ------------------------------------------------------------
def status_panel():
    st.markdown("### 🟢 Status Panel")
    col1, col2, col3 = st.columns(3)
    last_refresh = st.session_state.last_update
    latest_ts = st.session_state.get("_latest_ts")
    col1.metric("Polling", "ON ✅" if st.session_state.polling else "OFF ⛔")
    col2.metric("Last refresh", f"{last_refresh:%H:%M:%S} UTC" if last_refresh else "—")
    col3.metric("Latest Article", latest_ts.strftime("%Y-%m-%d %H:%M:%S") if latest_ts else "—")

# --- Top words/topics chart --------------------------------------------------
def plot_top_words(df: pd.DataFrame, text_cols: list[str], top_n: int = 15):
    if df.empty:
        return None
    import re
    from collections import Counter

    text = " ".join(df[col].dropna().astype(str) for col in text_cols if col in df.columns)
    words = re.findall(r"\b\w+\b", text.lower())

    stopwords = set([
        "the","and","for","with","this","that","from","are","was","will","have",
        "inc","corp","company","amazon","google","microsoft","apple","said"
    ])
    words = [w for w in words if len(w) > 3 and w not in stopwords]
    freq = Counter(words).most_common(top_n)
    if not freq:
        return None

    freq_df = pd.DataFrame(freq, columns=["word", "count"])
    fig = px.bar(freq_df.sort_values("count", ascending=True),
                 x="count", y="word", orientation="h",
                 title="Top Words in Headlines/Summaries")
    return fig

# --- App ---------------------------------------------------------------------
def main() -> None:
    init_state()

    # Polling toggle & conditional auto-refresh
    c1, c2, c3 = st.columns([1, 1, 1])
    last = st.session_state.last_update
    c1.caption(f"Last refresh: {last:%H:%M:%S} UTC" if last else "Last refresh: —")
    c2.caption("Polling: ✅" if st.session_state.polling else "Polling: ⛔")
    if c3.button("🔁 Toggle Polling", use_container_width=True):
        st.session_state.polling = not st.session_state.polling
        st.toast("Polling started" if st.session_state.polling else "Polling stopped")
        log_event("poll_toggle", {"on": st.session_state.polling})
        st.experimental_rerun()
    if st.session_state.polling:
        st_autorefresh(interval=POLL_INTERVAL * 1000, key="poller")

    st.title("Sentiment Analysis")
    st.divider()

    # Sidebar scope & filters
    with st.sidebar:
        st.header("Scope")
        st.session_state.mode = st.radio(
            "Analyse by", ["Company", "Topic", "Company + Topic"],
            index=["Company", "Topic", "Company + Topic"].index(st.session_state.mode),
        )

        preset_keys = list(DATE_PRESETS.keys())
        st.session_state.date_preset = st.selectbox(
            "Date range (UTC)", preset_keys,
            index=preset_keys.index(st.session_state.date_preset),
        )

        if st.session_state.mode in ["Company", "Company + Topic"]:
            st.session_state.company_input_mode = st.radio(
                "Company input", ["Pick from list", "Type manually"],
                index=0 if st.session_state.company_input_mode == "Pick from list" else 1,
            )
            if st.session_state.company_input_mode == "Pick from list":
                company_names = [c["name"] for c in COMPANIES]
                st.session_state.company_name = st.selectbox(
                    "Company", company_names,
                    index=company_names.index(st.session_state.company_name),
                )
            else:
                st.session_state.company_name = st.text_input(
                    "Company (name or ticker)", value=st.session_state.company_name,
                )

        if st.session_state.mode in ["Topic", "Company + Topic"]:
            st.session_state.topic_name = st.text_input(
                "Topic", value=st.session_state.topic_name,
            )

        if st.button("🧠 Get Sentiment Analysis", use_container_width=True):
            fetch_and_update_news()

    # Poll on each run (only does work when polling is ON)
    maybe_poll()

    # Use current news (pre-filter)
    news = st.session_state.news or {}
    df_all = news.get("docs", pd.DataFrame())
    subtitle = news.get("info", {}).get("subtitle", "")

    # Status panel (always visible)
    status_panel()

    # Filters
    st.subheader("Filters")
    df = df_all.copy()

    # Source filter
    if "source_name" in df.columns:
        sources = sorted(df["source_name"].dropna().unique())
        selected_sources = st.multiselect("Source", options=sources, default=sources)
        if selected_sources:
            df = df[df["source_name"].isin(selected_sources)]

    # Sentiment bucket filter
    sentiment_bucket = st.selectbox("Sentiment bucket", ["All", "Positive", "Neutral", "Negative"])
    if sentiment_bucket != "All" and "sentiment" in df.columns:
        s = pd.to_numeric(df["sentiment"], errors="coerce")
        if sentiment_bucket == "Positive":
            df = df[s > 0.1]
        elif sentiment_bucket == "Negative":
            df = df[s < -0.1]
        else:
            df = df[s.between(-0.1, 0.1)]

    # Keyword filter (title/summary/content)
    keyword = st.text_input("Keyword search in headlines/snippets", "")
    if keyword:
        mask = pd.Series(False, index=df.index)
        for col in [c for c in df.columns if any(k in c.lower() for k in ["title", "summary", "content"])]:
            mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
        df = df[mask]

    # Display results (filtered df)
    if not df.empty:
        st.subheader(subtitle)

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

        cfg = {}
        if "url" in df.columns:
            cfg["url"] = st.column_config.LinkColumn(display_text="Link")
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=cfg)

        # Top words chart
        fig_words = plot_top_words(df, ["title", "summary"])
        if fig_words:
            st.plotly_chart(fig_words, use_container_width=True)
    else:
        st.info("No news yet. Choose a scope and click **Get Sentiment Analysis**.")

    # Executive summary (based on CURRENT FILTERED DF)
    if not df.empty:
        st.subheader("Executive Summary")

        # New: Button to refresh LLM summary using current filters
        refresh_summary = st.button("🔄 Refresh Executive Summary", use_container_width=True)

        if st.session_state.summary_stale or refresh_summary:
            prompt = (
                "Summarise the following news items.\n"
                "End with one line: **Sentiment Score:** <NUMBER between -1 and 1>\n"
            )
            client = get_oauth_token()  # adapt to what your generate_response expects
            with st.spinner("Summarising..."):
                resp = generate_response(client, df.to_json(orient="records"), prompt)
                content = resp.get("content", "")
                score = extract_sentiment_score(content)
                st.session_state.summary_text = content
                st.session_state.llm_sentiment = float(score) if score is not None else None
                st.session_state.summary_stale = False
            log_event("summary_generated", {
                "latest_ts": st.session_state["_latest_ts"].isoformat() if st.session_state["_latest_ts"] else None,
                "filtered_rows": int(len(df)),
            })

        if st.session_state.llm_sentiment is not None:
            st.metric("LLM Sentiment", f"{st.session_state.llm_sentiment:.2f}")
        st.markdown(st.session_state.summary_text)

    st.caption("© MAAS Execution Analytics")


if __name__ == "__main__":
    main()
