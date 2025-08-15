# app.py
# Streamlit Sentiment Analysis — patched with (#4, #5, #6, #9, #13) + usage logging

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────── Config ───────────────────────────────
st.set_page_config(page_title="Sentiment Analysis", page_icon="📰", layout="wide")

# ---- Logging (file) ----
def _init_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "usage.log"

    logger = logging.getLogger("sentiment_app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

LOGGER = _init_logger()

def log_event(event: str, payload: Optional[Dict[str, Any]] = None, level: str = "info"):
    data = {"event": event, **(payload or {})}
    msg = json.dumps(data, ensure_ascii=False)
    getattr(LOGGER, level.lower(), LOGGER.info)(msg)

log_event("app_started", {"version": "0.1.0"})

# ─────────────────────── Mock data + helpers (replace) ───────────────────────
COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
]
TICKER_BY_NAME = {c["name"].upper(): c["ticker"] for c in COMPANIES}
NAME_BY_TICKER = {c["ticker"].upper(): c["name"] for c in COMPANIES}

@st.cache_data(show_spinner=False)
def get_news_data(type_of_search: str, subject: str, date_preset: str) -> Dict[str, Any]:
    """Return {'docs': pd.DataFrame, 'info': {...}}. Replace with your real fetch."""
    import random, time
    rng = random.Random(hash((type_of_search, subject, date_preset)) & 0xFFFFFFFF)
    rows = rng.randint(15, 40)
    now = pd.Timestamp.utcnow()
    df = pd.DataFrame({
        "timestamp": [now - pd.Timedelta(minutes=5*i) for i in range(rows)],
        "headline": [f"{subject}: News item {i}" for i in range(rows)],
        "source_name": [rng.choice(["FT", "Bloomberg", "Reuters"]) for _ in range(rows)],
        "scope": [type_of_search] * rows,
        "sentiment": [max(-1, min(1, rng.gauss(0.05 if type_of_search=="COMPANY" else 0, 0.35))) for _ in range(rows)],
    })
    time.sleep(0.2)  # simulate latency
    return {"docs": df, "info": {"subject": subject, "type": type_of_search, "date_preset": date_preset}}

def convert_to_xml(df: pd.DataFrame) -> str:
    """Replace with your own serialiser for the LLM."""
    rows = []
    for _, r in df.iterrows():
        rows.append(f"<doc ts='{r['timestamp']}' source='{r.get('source_name','')}' "
                    f"sentiment='{r.get('sentiment', 0)}'><h>{r.get('headline','')}</h></doc>")
    return "<docs>" + "".join(rows) + "</docs>"

def generate_response(llm, xml: str, prompt: str) -> Dict[str, str]:
    """Call your LLM here. This stub returns deterministic JSON."""
    # parse crude mean for the stub
    import re, statistics
    sents = [float(x) for x in re.findall(r"sentiment='(-?\d+\.?\d*)'", xml)]
    sc = statistics.fmean(sents) if sents else 0.0
    content = json.dumps({
        "summary": f"Coverage focuses on {len(sents)} items. Average sentiment ≈ {sc:.2f}.",
        "sentiment_score": round(sc, 2)
    })
    return {"content": content}

llm = object()  # placeholder handle

# ─────────────────────────────── Sidebar ───────────────────────────────
with st.sidebar:
    st.header("Scope")

    # Live-updating toggles (no clearing)
    mode = st.radio("Analyse by", ["Company", "Topic"], horizontal=True, key="mode")

    if mode == "Company":
        st.session_state.setdefault("company_input_mode", "Pick from list")
        st.radio(
            "Company input",
            ["Pick from list", "Type manually"],
            key="company_input_mode",
        )

    with st.form("controls_form", clear_on_submit=False):
        preset = st.selectbox(
            "Date preset (UTC)",
            ["Today", "Yesterday", "Last 24 hours", "Last 7 days", "Last 30 days"],
            index=3, key="date_preset"
        )

        # Inputs driven by the radios above
        company_name = company_ticker = None
        topic_name = topic_code = None

        if mode == "Company":
            if st.session_state["company_input_mode"] == "Pick from list":
                choice = st.selectbox(
                    "Company",
                    options=COMPANIES,
                    format_func=lambda c: f"{c['name']} ({c['ticker']})",
                    key="company_select",
                )
                company_name, company_ticker = choice["name"], choice["ticker"]
            else:
                typed = st.text_area(
                    "Company (name or ticker)",
                    placeholder="e.g., Apple Inc. or AAPL",
                    key="company_text", height=80
                ).strip()
                company_name = TICKER_BY_NAME.get(typed.upper()) or typed or "Company"
                company_ticker = NAME_BY_TICKER.get(typed.upper()) or typed.upper() or "COMPANY"
        else:
            topic_name = st.text_input("Topic", key="topic_name").strip()
            topic_code = (topic_name or "TOPIC").upper().replace(" ", "_")

        st.markdown("---")
        generate = st.form_submit_button("🧠 Get Sentiment Analysis", use_container_width=True)
        
st.title("Sentiment Analysis")

# ─────────────────────────────── Generate ───────────────────────────────
if generate:
    with st.spinner(text="Fetching news...", show_time=True):
        log_event("generate_clicked", {
            "mode": mode,
            "company_input_mode": st.session_state.get("company_input_mode"),
            "company": company_name if mode == "Company" else None,
            "company_code": company_ticker if mode == "Company" else None,
            "topic": topic_name if mode == "Topic" else None,
            "date_preset": preset,
        })

        # (#5) Topic must not be empty
        if mode == "Topic" and not (topic_name or "").strip():
            st.warning("Please enter a topic to generate news.")
            log_event("topic_missing", {})
            st.stop()

        frames = []
        if mode == "Company" and company_name:
            news = get_news_data(type_of_search="COMPANY",
                                 subject=company_name,
                                 date_preset=preset)
            frames.append(news["docs"])
        elif mode == "Topic":
            news = get_news_data(type_of_search="TOPIC",
                                 subject=topic_name,
                                 date_preset=preset)
            frames.append(news["docs"])

        df_all = pd.concat(frames, ignore_index=True).sort_values(by="timestamp", ascending=False) if frames else pd.DataFrame()
        st.session_state["news"] = {
            "docs": df_all,
            "info": {
                "mode": mode,
                "subject": company_name if mode == "Company" else topic_name,
                "subject_code": company_ticker if mode == "Company" else topic_code,
                "date_preset": preset,
            },
        }
        log_event("news_fetched", {"rows": int(len(df_all))})

# ─────────────────────────────── Main view ───────────────────────────────
news = st.session_state.get("news")
view_df = (news or {}).get("docs")
view_df = view_df if isinstance(view_df, pd.DataFrame) else pd.DataFrame()
subject_display = (news or {}).get("info", {}).get("subject", "")

if not view_df.empty:
    # KPIs (filled after compute)
    col1, col2, col3 = st.columns(3)
    col1.metric("Items", len(view_df))
    col2.metric("Avg sentiment", f"{view_df['sentiment'].mean():.2f}")
    col3.metric("Sentiment Std Dev", f"{view_df['sentiment'].std():.2f}")

    # (#6) Sentiment overview line with mean marker
    fig = go.Figure()
    fig.add_hline(y=0)
    mu = float(view_df["sentiment"].mean())
    fig.add_hline(y=mu, line_dash="dot")
    fig.add_trace(go.Scatter(x=[0], y=[mu], mode="markers", name="Average sentiment", marker=dict(size=10)))
    fig.update_layout(
        title="Sentiment Score",
        xaxis=dict(visible=False),
        yaxis=dict(range=[-1.1, 1.1], tickvals=[-1, -0.5, 0, 0.5, 1]),
        showlegend=False, height=180,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution + Box
    scopes = view_df["scope"].unique().tolist() if "scope" in view_df.columns else []
    color_col = "scope" if len(scopes) > 1 else None

    c1, c2 = st.columns(2)
    c1.subheader("Sentiment Score Distribution")
    fig_hist = px.histogram(view_df, x="sentiment", nbins=30, marginal="rug", color=color_col)
    c1.plotly_chart(fig_hist, use_container_width=True)

    c2.subheader("Sentiment Box Plot")
    fig_box = px.box(view_df, y="sentiment", points="all", color=color_col)
    c2.plotly_chart(fig_box, use_container_width=True)

    st.subheader("News Items")
    st.dataframe(view_df, use_container_width=True, hide_index=True)

    # Download
    if st.download_button("Download CSV", data=view_df.to_csv(index=False), file_name="news.csv"):
        log_event("csv_downloaded", {"rows": int(len(view_df))})

# ─────────────────────────────── LLM summary (#4 + #13) ───────────────────────────────
st.subheader(f"Executive Summary (LLM) — {subject_display or 'Selection'}")

docs_df = view_df  # already resolved above
if docs_df.empty:
    st.info("No news to summarise yet. Choose **Company** or **Topic**, then click **Get Sentiment Analysis**.")
else:
    default_prompt = """
You are an assistant creating an executive-style sentiment summary from article chunks.
Return ONLY a JSON object with keys:
{"summary": "...", "sentiment_score": <number between -1 and 1>}
Do not include any other text.
"""
    with st.spinner(text="Summarising...", show_time=True):
        try:
            xml_str = convert_to_xml(docs_df)
            resp = generate_response(llm, xml_str, default_prompt)   # must return {"content": "..."}
            obj = json.loads(resp["content"])
            summary = obj.get("summary", "")
            score = float(obj.get("sentiment_score", 0.0))

            st.session_state["response"] = resp
            st.session_state["summary_text"] = summary

            st.metric("LLM Sentiment", f"{score:.2f}")
            st.markdown(summary)

            log_event("llm_summary_ok", {"sentiment_score": score, "chars": len(summary)})
        except Exception as e:
            st.error(f"Could not generate summary: {e}")
            log_event("llm_summary_error", {"error": str(e)}, level="error")







# # ── Sentiment over time (colour coded) ──
# if not view_df.empty:
#     # ensure proper types & order
#     view_df = view_df.copy()
#     view_df["timestamp"] = pd.to_datetime(view_df["timestamp"], utc=True, errors="coerce")
#     view_df = view_df.dropna(subset=["timestamp", "sentiment"]).sort_values("timestamp")

#     # rolling mean (tune window as you like)
#     view_df["sent_roll"] = view_df["sentiment"].rolling(window=20, min_periods=3).mean()

#     fig_ts = go.Figure()

#     if st.session_state.get("colour_mode") == "Continuous":
#         # scatter with continuous diverging colour
#         fig_ts.add_trace(go.Scattergl(
#             x=view_df["timestamp"], y=view_df["sentiment"],
#             mode="markers", name="Items",
#             marker=dict(size=6, color=view_df["sentiment"], colorbar=dict(title="Sentiment"),
#                         colorscale="RdBu", reversescale=True, opacity=0.75),
#             hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Sent: %{y:.2f}<br>%{text}<extra></extra>",
#             text=view_df.get("headline", "")
#         ))
#     else:
#         # categorical colours by sign
#         def label(x: float) -> str:
#             return "Positive" if x > 0.10 else "Negative" if x < -0.10 else "Neutral"
#         view_df["sent_label"] = view_df["sentiment"].apply(label)

#         colour_map = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
#         for lab, g in view_df.groupby("sent_label", sort=False):
#             fig_ts.add_trace(go.Scattergl(
#                 x=g["timestamp"], y=g["sentiment"], mode="markers",
#                 name=lab, marker=dict(size=6, color=colour_map.get(lab, "#7f7f7f"), opacity=0.75),
#                 hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Sent: %{y:.2f}<br>%{text}<extra></extra>",
#                 text=g.get("headline", "")
#             ))

#     # rolling mean line
#     fig_ts.add_trace(go.Scatter(
#         x=view_df["timestamp"], y=view_df["sent_roll"],
#         mode="lines", name="Rolling mean (20)", line=dict(width=2)
#     ))

#     # zero line
#     fig_ts.add_hline(y=0, line_color="gray", opacity=0.6)

#     fig_ts.update_layout(
#         title="Sentiment Over Time",
#         xaxis=dict(title="Time", showspikes=True, spikemode="across"),
#         yaxis=dict(title="Sentiment", range=[-1.1, 1.1], tickvals=[-1, -0.5, 0, 0.5, 1]),
#         height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
#     )

#     st.plotly_chart(fig_ts, use_container_width=True)



import re, json

raw = resp["content"].strip()

# Remove leading/trailing code fences if present
if raw.startswith("```"):
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw.strip()).strip()

obj = json.loads(raw)



import re
def extract_sentiment_score(text):
    # Use regex to find the number after "Sentiment Score"
    match = re.search(r'\*\*Sentiment Score\*\* : \*\*([0-9.]+)\*\*', text)
    if match:
        return float(match.group(1))
    return None
