# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st

# --- External helpers you said live elsewhere -------------------------------
# Provide these in your own module.
# - get_oauth_token() -> str
# - get_news_data(type_of_search: str, search_string: str, date_preset: str) -> Dict[str, Any]
# - generate_response(llm_client, xml_str_or_text: str, system_prompt: str) -> Dict[str, str]  # returns {"content": "..."}
# - extract_sentiment_score(llm_content: str) -> Optional[float]
from services import (
    get_oauth_token,
    get_news_data,
    generate_response,
    extract_sentiment_score,
)

# --- Page config -------------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

# --- Constants ---------------------------------------------------------------
COMPANIES: List[Dict[str, str]] = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
]

# Your presets – keys are what the user sees, values are your backend enums/strings.
date_preset_dict: Dict[str, str] = {
    "today": "TODAY",
    "yesterday": "YESTERDAY",
    "last_1_hour": "LAST_ONE_HOUR",
    "last_3_hours": "LAST_THREE_HOURS",
    "last_6_hours": "LAST_SIX_HOURS",
    "last_12_hours": "LAST_TWELVE_HOURS",
    "last_24_hours": "LAST_TWENTY_FOUR_HOURS",
    "this_week": "THIS_WEEK",
    "last_week": "LAST_WEEK",
    "last_seven_days": "LAST_SEVEN_DAYS",
    "last_thirty_days": "LAST_THIRTY_DAYS",
    "year_to_date": "YEAR_TO_DATE",
    "last_twelve_months": "LAST_YEAR",
}

# Normalised maps for robust user input handling
def _norm(s: str) -> str:
    return s.strip().casefold()

TICKER_BY_NAME = {_norm(c["name"]): c["ticker"] for c in COMPANIES}
NAME_BY_TICKER = {_norm(c["ticker"]): c["name"] for c in COMPANIES}
COMPANY_OPTIONS = [f"{c['name']} ({c['ticker']})" for c in COMPANIES]

# --- Caching: client + data --------------------------------------------------
@st.cache_resource
def get_llm_client():
    # If you also need base_url, add it in your services.get_oauth_token() or here.
    # This function should construct and return your LLM client instance.
    # Keep it cached so we don't recreate it every rerun.
    _ = get_oauth_token()  # Ensure token available/valid; your client may need it implicitly.
    # Example if you use openai:
    # from openai import OpenAI
    # return OpenAI(api_key=_)
    return object()  # placeholder; your services.generate_response uses what it needs

@st.cache_data(ttl=60, show_spinner=False)
def get_news_cached(search_type: str, query: str, preset_key: str) -> Dict[str, Any]:
    """Cache API calls for 60s to avoid hammering the backend while polling."""
    backend_preset = date_preset_dict[preset_key]
    return get_news_data(type_of_search=search_type, search_string=query, date_preset=backend_preset)

# --- Utilities ---------------------------------------------------------------
def init_logger() -> None:
    """Very small file logger for usage stats."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    st.session_state.setdefault("_log_file", log_dir / f"usage-{datetime.today():%Y%m%d}.log")

def log_event(event: str, payload: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
    try:
        file: Path = st.session_state["_log_file"]
        data = {"ts": datetime.utcnow().isoformat(), "event": event, "payload": payload or {}}
        with file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        # Don't break the app on logging problems
        pass

def colour_for_score(score: float) -> str:
    """Return a hex colour from a diverging colormap based on score in [-1, 1]."""
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = mcolors.get_cmap("RdYlGn")
    rgba = cmap(norm(score))
    return mcolors.to_hex(rgba)

def split_company_label(label: str) -> tuple[str, str]:
    """'Apple Inc. (AAPL)' -> ('Apple Inc.', 'AAPL')"""
    if "(" in label and label.endswith(")"):
        name = label[: label.rfind("(")].strip()
        ticker = label[label.rfind("(") + 1 : -1].strip()
        return name, ticker
    return label, ""

def resolve_company(user_typed: str) -> tuple[Optional[str], Optional[str]]:
    """Return (name, ticker) trying both directions, robust to case/space."""
    n = _norm(user_typed)
    ticker = TICKER_BY_NAME.get(n) or user_typed.strip().upper() if _norm(user_typed) in NAME_BY_TICKER else None
    if not ticker:
        ticker = NAME_BY_TICKER.get(n) and user_typed.strip().upper()
    name = NAME_BY_TICKER.get(_norm(ticker or "")) or TICKER_BY_NAME.get(n) and user_typed.strip()
    return (name, ticker)

# --- Session state (single source of truth) ----------------------------------
def init_state() -> None:
    st.session_state.setdefault("mode", "Company")              # "Company" or "Topic"
    st.session_state.setdefault("company_input_mode", "Pick from list")  # or "Type manually"
    st.session_state.setdefault("date_preset", "this_week")

    st.session_state.setdefault("polling", False)
    st.session_state.setdefault("next_poll_at", None)

    # News container:
    # st.session_state["news"] = {"docs": pd.DataFrame, "info": {...}}
    st.session_state.setdefault("news", None)

    # LLM outputs
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)

    # UI placeholders
    st.session_state.setdefault("summary_ph", None)
    st.session_state.setdefault("sentiment_line_ph", None)
    st.session_state.setdefault("last_update", None)

# --- Polling (non-blocking) --------------------------------------------------
def fetch_and_update_news() -> None:
    """Fetch and merge news, update state, no blocking."""
    mode = st.session_state.mode
    preset = st.session_state.date_preset

    if mode == "Company":
        subject = st.session_state.get("company_name")
        if not subject:
            return
        res = get_news_cached("COMPANY", subject, preset)
        subtitle = f"Company Details – {res.get('info', {}).get('name','')}"
    else:
        subject = st.session_state.get("topic_name")
        if not subject:
            return
        res = get_news_cached("TOPIC", subject, preset)
        subtitle = f"Topic – {res.get('info', {}).get('name','')}"

    df = res.get("res") if isinstance(res, dict) else None
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    st.session_state["news"] = {
        "docs": df.sort_values("timestamp", ascending=False).reset_index(drop=True),
        "info": {
            "mode": mode,
            "subject": subject or "",
            "date_preset": preset,
            "subtitle": subtitle,
        },
    }
    st.session_state["last_update"] = datetime.utcnow()
    log_event("news_fetched", {"rows": int(len(df))})

def maybe_poll() -> None:
    if not st.session_state.polling:
        return
    now = datetime.utcnow()
    due = st.session_state.next_poll_at is None or now >= st.session_state.next_poll_at
    if due:
        fetch_and_update_news()
        st.session_state.next_poll_at = now + timedelta(seconds=60)
        st.toast("Polled for updates")

# --- UI helpers --------------------------------------------------------------
def top_status_row() -> None:
    c1, c2, c3 = st.columns([1, 1, 2])
    last = st.session_state.last_update
    c1.caption(f"Last refresh: {last:%H:%M:%S} UTC" if last else "Last refresh: —")
    c2.caption("Polling: ✅" if st.session_state.polling else "Polling: ⛔")
    if st.button("🔁 Start / Stop Updates", use_container_width=False):
        st.session_state.polling = not st.session_state.polling
        if st.session_state.polling:
            st.session_state.next_poll_at = datetime.utcnow()  # poll immediately
            st.toast("Polling started")
        else:
            st.toast("Polling stopped")
        st.rerun()

def sentiment_scale(avg: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1, 1], y=[0, 0], mode="lines",
                             line=dict(color="gray", width=2), name="Sentiment Scale"))
    fig.add_trace(go.Scatter(x=[avg], y=[0], mode="markers",
                             marker=dict(color=colour_for_score(avg), size=15),
                             name="Average Sentiment"))
    fig.update_layout(
        title="Sentiment Score",
        xaxis=dict(
            range=[-1, 1],
            tickvals=[-1, 0, 1],
            ticktext=["−1 (Very negative)", "0 (Neutral)", "+1 (Very positive)"],
            zeroline=True, zerolinewidth=4, zerolinecolor="gray",
        ),
        yaxis=dict(visible=False),
        showlegend=False,
        height=180,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# --- App ---------------------------------------------------------------------
def main() -> None:
    init_logger()
    init_state()
    maybe_poll()  # non-blocking periodic work

    st.title("Sentiment Analysis")
    top_status_row()
    st.divider()

    # --- Controls (radios live, submit button in form) ----------------------
    with st.sidebar:
        st.header("Scope")

        # Live radios (outside the form so they update immediately)
        st.session_state.mode = st.radio(
            "Analyse by",
            ["Company", "Topic"],
            horizontal=True,
            key="mode",
        )

        st.session_state.company_input_mode = st.radio(
            "Company input",
            ["Pick from list", "Type manually"],
            key="company_input_mode",
        ) if st.session_state.mode == "Company" else st.session_state.company_input_mode

        st.session_state.date_preset = st.selectbox(
            "Date preset (UTC)",
            list(date_preset_dict.keys()),
            index=list(date_preset_dict.keys()).index(st.session_state.date_preset),
            key="date_preset",
        )

        # Inputs driven by the radios above
        company_name = company_ticker = None
        topic_name = None

        if st.session_state.mode == "Company":
            if st.session_state.company_input_mode == "Pick from list":
                choice = st.selectbox("Company", options=COMPANY_OPTIONS, key="company_select")
                company_name, company_ticker = split_company_label(choice)
            else:
                typed = st.text_area("Company (name or ticker)", height=80, key="company_text").strip()
                if typed:
                    name, tick = resolve_company(typed)
                    company_name, company_ticker = name, tick
            # store for polling
            st.session_state["company_name"] = company_name
            st.session_state["company_ticker"] = company_ticker
        else:
            topic_name = st.text_input("Topic", key="topic_name").strip()
            st.session_state["topic_name"] = topic_name

        # Submit only wraps the inputs that should trigger a fetch
        with st.form("controls_form", clear_on_submit=False):
            generate = st.form_submit_button("🧠 Get Sentiment Analysis", use_container_width=True)

    # --- Generate on click or first run --------------------------------------
    if generate or st.session_state.get("news") is None:
        fetch_and_update_news()

    news = st.session_state.get("news") or {}
    view_df: pd.DataFrame = news.get("docs", pd.DataFrame())
    info: Dict[str, Any] = news.get("info", {})
    subtitle = info.get("subtitle") or ""

    if not view_df.empty:
        st.subheader(subtitle)

        # KPIs
        s_col1, s_col2 = st.columns(2)
        if "sentiment" in view_df:
            s_series = pd.to_numeric(view_df["sentiment"], errors="coerce").dropna()
            avg = float(s_series.mean()) if not s_series.empty else 0.0
            std = float(s_series.std()) if not s_series.empty else 0.0
        else:
            avg, std = 0.0, 0.0

        s_col1.metric("Avg Sentiment", f"{avg:.2f}")
        s_col2.metric("Sentiment Std Dev", f"{std:.2f}")

        # Sentiment scale
        fig_scale = sentiment_scale(avg)
        st.plotly_chart(fig_scale, use_container_width=True)

        # Distribution + box
        c1, c2 = st.columns(2)
        if "sentiment" in view_df:
            fig_hist = px.histogram(view_df, x="sentiment", nbins=30, marginal="rug",
                                    color="source_name", title="Sentiment Score Distribution")
            c1.plotly_chart(fig_hist, use_container_width=True)

            fig_box = px.box(view_df, y="sentiment", points="all", color="source_name",
                             title="Sentiment Box Plot")
            c2.plotly_chart(fig_box, use_container_width=True)

        # Sentiment over time
        if "timestamp" in view_df and "sentiment" in view_df:
            ts_df = view_df.sort_values("timestamp", ascending=True).copy()
            ts_df["moving_avg_sentiment"] = pd.to_numeric(ts_df["sentiment"], errors="coerce").expanding().mean()

            fig_ts = px.scatter(ts_df, x="timestamp", y="sentiment",
                                hover_data=[c for c in ts_df.columns if c not in {"timestamp", "sentiment"}])
            fig_ts.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["moving_avg_sentiment"],
                                        mode="lines", name="Moving Avg Sentiment"))
            fig_ts.add_hline(y=0, line_color="gray", opacity=0.6)
            fig_ts.update_layout(
                title="Sentiment Over Time",
                xaxis=dict(title="Time", showspikes=True, spikemode="across"),
                yaxis=dict(title="Sentiment", range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
                height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # News table
        st.subheader(f"{len(view_df)} News Items:")
        st.dataframe(
            view_df,
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn(display_text="Link")},
        )
    else:
        st.info("No news to summarise yet. Choose a Company or Topic, then click **Get Sentiment Analysis**.")

    # --- Executive summary via LLM -------------------------------------------
    if not view_df.empty:
        st.subheader("Executive Summary")
        default_prompt = (
            "You are a content summarising assistant.\n"
            "You will be given text chunks about a company/topic.\n"
            "Create a concise executive summary and include a single line:\n"
            "**Sentiment Score:** <NUMBER>\n"
            "The score must be between -1 and 1 and on its own line exactly as shown.\n"
            "Only use the provided information.\n"
        )

        # Use your own conversion if needed; we’ll simply pass the table as JSON here.
        chunks_as_text = view_df.to_json(orient="records")

        with st.spinner("Summarising..."):
            llm_client = get_llm_client()
            llm_resp = generate_response(llm_client, chunks_as_text, default_prompt)  # -> {"content": "..."}
            content = llm_resp.get("content", "")

            score = extract_sentiment_score(content)
            st.session_state["summary_text"] = content
            st.session_state["llm_sentiment"] = float(score) if score is not None else None

        # Render
        st.metric("LLM Sentiment", f"{st.session_state['llm_sentiment']:.2f}" if st.session_state["llm_sentiment"] is not None else "—")
        st.markdown(st.session_state["summary_text"])

    # --- Footer ---------------------------------------------------------------
    st.caption("© MAAS Execution Analytics")

if __name__ == "__main__":
    main()
