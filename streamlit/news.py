# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st

# --- External helpers you said live elsewhere -------------------------------
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

# Polling and cache constants
POLL_INTERVAL_SECONDS = 60
CACHE_TTL_SECONDS = 60

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
@lru_cache(maxsize=128)
def _norm(s: str) -> str:
    """Cached string normalization for efficiency."""
    if not isinstance(s, str):
        return ""
    return s.strip().casefold()

TICKER_BY_NAME = {_norm(c["name"]): c["ticker"] for c in COMPANIES}
NAME_BY_TICKER = {_norm(c["ticker"]): c["name"] for c in COMPANIES}
COMPANY_OPTIONS = [f"{c['name']} ({c['ticker']})" for c in COMPANIES]

# --- Caching: client + data --------------------------------------------------
@st.cache_resource
def get_llm_client():
    """Get cached LLM client instance."""
    try:
        token = get_oauth_token()
        class MockLLMClient:
            def __init__(self, token: str):
                self.token = token
        return MockLLMClient(token)
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_news_cached(search_type: str, query: str, preset_key: str) -> Dict[str, Any]:
    """Cache API calls to avoid hammering the backend while polling."""
    if not query or not query.strip():
        return {"res": pd.DataFrame(), "info": {}}
    
    backend_preset = date_preset_dict.get(preset_key, "THIS_WEEK")
    try:
        return get_news_data(type_of_search=search_type, search_string=query.strip(), date_preset=backend_preset)
    except Exception as e:
        st.error(f"Failed to fetch news data: {e}")
        return {"res": pd.DataFrame(), "info": {}}

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
        pass

def colour_for_score(score: float) -> str:
    """Return a hex colour from a diverging colormap based on score in [-1, 1]."""
    score = max(-1.0, min(1.0, score))
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = mcolors.get_cmap("RdYlGn")
    rgba = cmap(norm(score))
    return mcolors.to_hex(rgba)

def split_company_label(label: str) -> Tuple[str, str]:
    """'Apple Inc. (AAPL)' -> ('Apple Inc.', 'AAPL')"""
    if not isinstance(label, str):
        return "", ""
    
    if "(" in label and label.endswith(")"):
        name = label[: label.rfind("(")].strip()
        ticker = label[label.rfind("(") + 1 : -1].strip()
        return name, ticker
    return label, ""

def resolve_company(user_typed: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (name, ticker) trying both directions, robust to case/space."""
    if not isinstance(user_typed, str) or not user_typed.strip():
        return None, None
    
    normalized_input = _norm(user_typed)
    
    # Try to find ticker by company name
    if normalized_input in TICKER_BY_NAME:
        ticker = TICKER_BY_NAME[normalized_input]
        name = user_typed.strip()
        return name, ticker
    
    # Try to find name by ticker
    if normalized_input in NAME_BY_TICKER:
        name = NAME_BY_TICKER[normalized_input]
        ticker = user_typed.strip().upper()
        return name, ticker
    
    # If not found, treat as manual input
    cleaned_input = user_typed.strip()
    if len(cleaned_input) <= 5 and cleaned_input.isupper():
        return None, cleaned_input  # Treat as ticker
    else:
        return cleaned_input, None  # Treat as company name

# --- Session state -----------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("mode", "Company")
    st.session_state.setdefault("company_input_mode", "Pick from list")
    st.session_state.setdefault("date_preset", "this_week")
    st.session_state.setdefault("polling", False)
    st.session_state.setdefault("next_poll_at", None)
    st.session_state.setdefault("news", None)
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)
    st.session_state.setdefault("last_update", None)

# --- Polling ------------------------------------------------------------------
def fetch_and_update_news() -> None:
    """Fetch and merge news, update state, no blocking."""
    mode = st.session_state.mode
    preset = st.session_state.date_preset

    if mode == "Company":
        subject = st.session_state.get("company_name")
        if not subject or not subject.strip():
            st.warning("Please select or enter a company name.")
            return
        res = get_news_cached("COMPANY", subject, preset)
        subtitle = f"Company Details – {res.get('info', {}).get('name', subject)}"
    else:
        subject = st.session_state.get("topic_name")
        if not subject or not subject.strip():
            st.warning("Please enter a topic.")
            return
        res = get_news_cached("TOPIC", subject, preset)
        subtitle = f"Topic – {res.get('info', {}).get('name', subject)}"

    df = res.get("res") if isinstance(res, dict) else None
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    # Ensure sentiment column is properly processed
    if not df.empty and "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

    st.session_state["news"] = {
        "docs": df.sort_values("timestamp", ascending=False).reset_index(drop=True) if not df.empty and "timestamp" in df.columns else df,
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
    """Handle polling - checks if it's time to poll on each page interaction."""
    if not st.session_state.polling:
        return
    
    now = datetime.utcnow()
    due = st.session_state.next_poll_at is None or now >= st.session_state.next_poll_at
    
    if due:
        fetch_and_update_news()
        st.session_state.next_poll_at = now + timedelta(seconds=POLL_INTERVAL_SECONDS)
        st.toast("Polled for updates")

def add_auto_refresh() -> None:
    """Add JavaScript-based auto-refresh when polling is enabled."""
    if st.session_state.polling:
        st.markdown(
            """
            <script>
            setTimeout(function(){
                window.location.reload(1);
            }, 30000);
            </script>
            """,
            unsafe_allow_html=True
        )

# --- UI helpers --------------------------------------------------------------
def top_status_row() -> None:
    c1, c2, c3 = st.columns([1, 1, 1])
    last = st.session_state.last_update
    c1.caption(f"Last refresh: {last:%H:%M:%S} UTC" if last else "Last refresh: —")
    
    # Enhanced polling status
    if st.session_state.polling:
        next_poll = st.session_state.next_poll_at
        if next_poll:
            seconds_until = (next_poll - datetime.utcnow()).total_seconds()
            if seconds_until > 0:
                c2.caption(f"Polling: ✅ (next in {int(seconds_until)}s)")
            else:
                c2.caption("Polling: ✅ (due now)")
        else:
            c2.caption("Polling: ✅ (starting)")
    else:
        c2.caption("Polling: ⛔")
    
    with c3:
        if st.button("🔁 Start / Stop Updates", use_container_width=True):
            st.session_state.polling = not st.session_state.polling
            if st.session_state.polling:
                st.session_state.next_poll_at = datetime.utcnow()
                st.toast("Polling started")
            else:
                st.toast("Polling stopped")
            st.rerun()

def sentiment_scale(avg: float) -> go.Figure:
    if not isinstance(avg, (int, float)) or pd.isna(avg):
        avg = 0.0
    avg = max(-1.0, min(1.0, avg))
    
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

# --- Main App ----------------------------------------------------------------
def main() -> None:
    init_logger()
    init_state()
    maybe_poll()
    add_auto_refresh()

    st.title("Sentiment Analysis")
    top_status_row()
    st.divider()

    # --- Sidebar Controls (COMPLETELY FIXED) --------------------------------
    with st.sidebar:
        st.header("Scope")

        # Get current session state values for defaults
        current_mode = st.session_state.get("mode", "Company")
        current_company_mode = st.session_state.get("company_input_mode", "Pick from list")
        current_preset = st.session_state.get("date_preset", "this_week")

        # Create widgets with unique keys
        selected_mode = st.radio(
            "Analyse by",
            ["Company", "Topic"],
            index=0 if current_mode == "Company" else 1,
            horizontal=True,
            key="analysis_mode_widget"
        )

        if selected_mode == "Company":
            selected_company_mode = st.radio(
                "Company input",
                ["Pick from list", "Type manually"],
                index=0 if current_company_mode == "Pick from list" else 1,
                key="company_input_mode_widget"
            )
        else:
            selected_company_mode = current_company_mode

        selected_preset = st.selectbox(
            "Date preset (UTC)",
            list(date_preset_dict.keys()),
            index=list(date_preset_dict.keys()).index(current_preset),
            key="date_preset_widget"
        )

        # Update session state
        st.session_state.mode = selected_mode
        st.session_state.company_input_mode = selected_company_mode
        st.session_state.date_preset = selected_preset

        # --- FORM SECTION ----------------------------------------------------
        with st.form("analysis_form"):
            company_name = company_ticker = topic_name = None

            if selected_mode == "Company":
                if selected_company_mode == "Pick from list":
                    # Default selection based on current company
                    current_company = st.session_state.get("company_name", "")
                    default_idx = 0
                    for i, option in enumerate(COMPANY_OPTIONS):
                        if current_company and current_company in option:
                            default_idx = i
                            break

                    selected_company = st.selectbox(
                        "Company",
                        options=COMPANY_OPTIONS,
                        index=default_idx,
                        key="form_company_dropdown"
                    )
                    company_name, company_ticker = split_company_label(selected_company)
                else:
                    # Manual company input
                    current_value = st.session_state.get("company_name", "") or st.session_state.get("company_ticker", "")
                    company_input = st.text_area(
                        "Company (name or ticker)",
                        value=current_value,
                        height=80,
                        key="form_company_manual"
                    )
                    if company_input.strip():
                        company_name, company_ticker = resolve_company(company_input.strip())
            else:
                # Topic input
                current_topic = st.session_state.get("topic_name", "")
                topic_input = st.text_input(
                    "Topic",
                    value=current_topic,
                    key="form_topic_input"
                )
                topic_name = topic_input.strip() if topic_input else None

            # Form submit
            submitted = st.form_submit_button("🧠 Get Sentiment Analysis", use_container_width=True)

            # Update session state only on form submit
            if submitted:
                if selected_mode == "Company":
                    st.session_state["company_name"] = company_name or company_ticker or ""
                    st.session_state["company_ticker"] = company_ticker or ""
                else:
                    st.session_state["topic_name"] = topic_name or ""

    # --- Manual refresh button ----------------------------------------------
    col1, col2 = st.columns([1, 4])
    with col1:
        refresh_clicked = st.button("🔄 Refresh Data")

    # --- Data fetching logic -------------------------------------------------
    # Auto-load default on first visit
    if st.session_state.get("news") is None and not submitted and not refresh_clicked:
        if selected_mode == "Company" and not st.session_state.get("company_name"):
            default_company = COMPANIES[0]
            st.session_state["company_name"] = default_company["name"]
            st.session_state["company_ticker"] = default_company["ticker"]

    # Fetch data
    if submitted or refresh_clicked:
        fetch_and_update_news()
    elif st.session_state.get("news") is None and st.session_state.get("company_name"):
        fetch_and_update_news()

    # --- Display results -----------------------------------------------------
    news = st.session_state.get("news") or {}
    view_df: pd.DataFrame = news.get("docs", pd.DataFrame())
    info: Dict[str, Any] = news.get("info", {})
    subtitle = info.get("subtitle") or ""

    if not view_df.empty:
        st.subheader(subtitle)

        # KPIs with proper error handling
        s_col1, s_col2 = st.columns(2)
        
        if "sentiment" in view_df.columns:
            s_series = pd.to_numeric(view_df["sentiment"], errors="coerce").dropna()
            if not s_series.empty:
                avg = float(s_series.mean())
                std = float(s_series.std())
            else:
                avg, std = 0.0, 0.0
        else:
            avg, std = 0.0, 0.0
            st.warning("Sentiment data not available in the dataset.")

        s_col1.metric("Avg Sentiment", f"{avg:.2f}")
        s_col2.metric("Sentiment Std Dev", f"{std:.2f}")

        # Sentiment scale
        fig_scale = sentiment_scale(avg)
        st.plotly_chart(fig_scale, use_container_width=True)

        # Distribution + box (only if sentiment data exists)
        if "sentiment" in view_df.columns and not view_df["sentiment"].isna().all():
            c1, c2 = st.columns(2)
            
            fig_hist = px.histogram(view_df, x="sentiment", nbins=30, marginal="rug",
                                    color="source_name" if "source_name" in view_df.columns else None, 
                                    title="Sentiment Score Distribution")
            c1.plotly_chart(fig_hist, use_container_width=True)

            fig_box = px.box(view_df, y="sentiment", points="all", 
                             color="source_name" if "source_name" in view_df.columns else None,
                             title="Sentiment Box Plot")
            c2.plotly_chart(fig_box, use_container_width=True)

        # Sentiment over time
        if "timestamp" in view_df.columns and "sentiment" in view_df.columns and not view_df["sentiment"].isna().all():
            ts_df = view_df.sort_values("timestamp", ascending=True).copy()
            ts_df["sentiment_numeric"] = pd.to_numeric(ts_df["sentiment"], errors="coerce")
            ts_df["moving_avg_sentiment"] = ts_df["sentiment_numeric"].expanding().mean()

            fig_ts = px.scatter(ts_df, x="timestamp", y="sentiment_numeric",
                                hover_data=[c for c in ts_df.columns if c not in {"timestamp", "sentiment", "sentiment_numeric"}])
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
            column_config={"url": st.column_config.LinkColumn(display_text="Link")} if "url" in view_df.columns else None,
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

        chunks_as_text = view_df.to_json(orient="records")

        llm_client = get_llm_client()
        if llm_client is not None:
            try:
                with st.spinner("Summarising..."):
                    llm_resp = generate_response(llm_client, chunks_as_text, default_prompt)
                    content = llm_resp.get("content", "")

                    score = extract_sentiment_score(content)
                    st.session_state["summary_text"] = content
                    st.session_state["llm_sentiment"] = float(score) if score is not None else None

                # Render
                if st.session_state["llm_sentiment"] is not None:
                    st.metric("LLM Sentiment", f"{st.session_state['llm_sentiment']:.2f}")
                else:
                    st.metric("LLM Sentiment", "—")
                st.markdown(st.session_state["summary_text"])
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
        else:
            st.error("LLM client not available. Cannot generate summary.")

    # --- Footer ---------------------------------------------------------------
    st.caption("© MAAS Execution Analytics")

if __name__ == "__main__":
    main()
