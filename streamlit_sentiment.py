# app.py
# Streamlit Sentiment Analysis POC (mock data)
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date, time as dtime
import random
import plotly.express as px
import hashlib

st.set_page_config(page_title="Sentiment Analysis POC", page_icon="📰", layout="wide")

# -------------------------------------
# Mock universe
# -------------------------------------
COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
    {"ticker": "META", "name": "Meta Platforms, Inc."},
    {"ticker": "TSLA", "name": "Tesla, Inc."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    {"ticker": "BARC", "name": "Barclays PLC"},
    {"ticker": "HSBA", "name": "HSBC Holdings plc"},
    {"ticker": "BP",   "name": "BP p.l.c."},
    {"ticker": "SHEL", "name": "Shell plc"},
    {"ticker": "AAL",  "name": "Anglo American plc"},
]
TICKER_BY_NAME = {c["name"]: c["ticker"] for c in COMPANIES}
NAME_BY_TICKER = {c["ticker"]: c["name"] for c in COMPANIES}
COMPANY_OPTIONS = [f"{c['name']} ({c['ticker']})" for c in COMPANIES]

# Fixed mock controls
N_ITEMS = 60
MU = 0.05
SIGMA = 0.35
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# --- Session defaults
for k, v in {
    "news_df": None,
    "subject_display": "",
    "subject_code": "SUBJ",
    "summary_text": "",
    "generated_at": None,
}.items():
    st.session_state.setdefault(k, v)

# -------------------------------------
# Helpers
# -------------------------------------
def stable_int_from_string(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

@st.cache_data(show_spinner=False)
def generate_mock_news(subject: str, code: str, kind: str,
                       n: int, start_dt_utc: datetime, end_dt_utc: datetime,
                       mu: float, sigma: float, seed_val: int) -> pd.DataFrame:
    """Generate mock articles for a company/topic between start/end (UTC)."""
    subj_seed = seed_val + stable_int_from_string(f"{kind}:{code}") % 10_000_000
    rng = np.random.default_rng(subj_seed)

    span_seconds = max(1, int((end_dt_utc - start_dt_utc).total_seconds()))
    rand_secs = rng.integers(0, span_seconds, size=n)
    times = [start_dt_utc + timedelta(seconds=int(s)) for s in rand_secs]
    times = sorted(times, reverse=True)

    sources = ["Bloomberg", "Reuters", "FT", "WSJ", "CNBC", "MarketWatch", "Morningstar", "City A.M.", "The Times"]
    verbs = ["tops", "misses", "guides", "warns", "launches", "expands", "settles", "wins", "partners", "invests in"]
    objects = [
        "Q earnings", "revenue estimates", "outlook", "new product line", "AI initiative", "share buyback",
        "regulatory probe", "data centre push", "UK expansion", "cost-cut programme"
    ]
    body_bits = [
        "Management commentary points to near-term variability but underscores medium-term growth drivers.",
        "Analysts remain split on the scope of margin expansion given input cost dynamics.",
        "Channel checks suggest stabilising demand in core segments with upside from new launches.",
        "Regulatory overhang persists but base-case timelines appear manageable.",
        "Valuation screens as fair vs. peers; catalysts include events over the next 1–4 weeks."
    ]

    sentiments = np.clip(rng.normal(loc=mu, scale=sigma, size=n), -1, 1)

    rows = []
    for i in range(n):
        src = random.choice(sources)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        headline = f"{subject} {verb} {obj}" if kind == "company" else f"{subject}: {verb} {obj}"
        url = f"https://example.com/{code}/{i}"
        s = float(sentiments[i])
        label = "Positive" if s > 0.15 else ("Negative" if s < -0.15 else "Neutral")
        content = (
            f"{headline}. {random.choice(body_bits)} Sentiment skew: {label.lower()} (score {s:.2f}). "
            f"{random.choice(body_bits)}"
        )
        rows.append({
            "datetime_utc": times[i].replace(microsecond=0, tzinfo=timezone.utc),
            "source": src,
            "subject": subject,
            "headline": headline,
            "url": url,
            "sentiment": s,
            "label": label,
            "content": content,
            "kind": kind,                                # "company" or "topic"
            "scope": f"{'Company' if kind=='company' else 'Topic'}: {subject}",
            "code": code,
        })

    df = pd.DataFrame(rows)
    df["datetime_local"] = pd.to_datetime(df["datetime_utc"]).dt.tz_convert(None)  # naive local for display
    return df.sort_values("datetime_utc", ascending=False)

def mock_llm_summary(df: pd.DataFrame, subject_display: str, prompt: str) -> str:
    avg = df["sentiment"].mean()
    pos = (df["label"] == "Positive").sum()
    neg = (df["label"] == "Negative").sum()
    neu = (df["label"] == "Neutral").sum()
    latest = df.sort_values("datetime_utc", ascending=False).head(5)
    top_pos = df.sort_values("sentiment", ascending=False).head(2)["headline"].tolist()
    top_neg = df.sort_values("sentiment", ascending=True).head(2)["headline"].tolist()
    tilt = "constructive" if avg > 0.05 else ("cautious" if avg < -0.05 else "balanced")

    lines = []
    lines.append(f"Prompt noted ({len(prompt)} chars). Below is a mock summary based on the dataset only.")
    lines.append("")
    lines.append(f"**Subject**: {subject_display}")
    lines.append(f"**Tone**: {tilt.capitalize()} | **Avg sentiment**: {avg:.2f} | **Split**: {pos}↑ / {neu}→ / {neg}↓")
    lines.append("")
    lines.append("**Key Positives**")
    for h in top_pos:
        lines.append(f"• {h}")
    lines.append("")
    lines.append("**Key Risks**")
    for h in top_neg:
        lines.append(f"• {h}")
    lines.append("")
    lines.append("**Recent Coverage (last 5 items)**")
    for _, r in latest.iterrows():
        ts = pd.to_datetime(r["datetime_utc"]).strftime("%d %b %H:%M UTC")
        lines.append(f"• [{ts}] {r['source']}: {r['headline']}")
    lines.append("")
    lines.append(f"**Bottom line**: Overall read is {tilt}; watch for near-term catalysts and guidance updates.")
    return "\n".join(lines)

# -------------------------------------
# Sidebar — Controls Form (prevents rerun until submit)
# -------------------------------------
# Sidebar — Controls (toggle outside form; inputs inside form)
# -------------------------------------
# -------------------------------------
# Sidebar — Controls (no "Both", presets only)
# -------------------------------------
with st.sidebar:
    st.header("🔎 Scope & Range")

    # Outside the form so the UI switches immediately when toggled
    mode = st.radio("Analyse by", ["Company", "Topic"], horizontal=True, key="mode")

    if mode == "Company":
        # Toggle outside the form so it reruns and swaps the widget
        st.session_state.setdefault("company_input_mode", "Pick from list")
        company_input_mode = st.radio(
            "Company input",
            ["Pick from list", "Type manually"],
            horizontal=False,
            key="company_input_mode",
        )

    # Rest in a form so it doesn’t rerun until submit
    with st.form(key="controls_form", clear_on_submit=False):
        company_name = company_ticker = None
        topic_name = topic_code = None

        # --- Company inputs (only for Company)
        if mode == "Company":
            if st.session_state["company_input_mode"] == "Pick from list":
                COMPANY_OPTIONS = [f"{c['name']} ({c['ticker']})" for c in COMPANIES]
                company_choice = st.selectbox("Company", COMPANY_OPTIONS, index=0, key="company_select")
                name = company_choice.split(" (")[0]
                code = company_choice.split("(")[-1].rstrip(")")
            else:
                typed = st.text_area(
                    "Company (name or ticker)",
                    placeholder="e.g., Apple Inc. or AAPL",
                    key="company_text",
                    height=80,
                ).strip()
                # Resolve to name/ticker if known; otherwise use literal
                code = typed.upper() if typed.upper() in NAME_BY_TICKER else None
                name = NAME_BY_TICKER.get(typed.upper()) or TICKER_BY_NAME.get(typed) or typed
                if code is None:
                    code = (typed.upper()[:8] or "COMPANY")
            company_name, company_ticker = name, code

        # --- Topic inputs (only for Topic)
        if mode == "Topic":
            topic_name = st.text_input("Topic", placeholder="e.g., AI in data centres", key="topic_name").strip()
            topic_code = (topic_name or "TOPIC").upper().replace(" ", "_")

        st.markdown("---")

        # Date presets only
        from datetime import datetime as dt, timedelta, timezone, time as dtime
        today_utc = dt.now(timezone.utc).date()
        preset = st.selectbox(
            "Date preset (UTC)",
            ["Today", "Yesterday", "Last 24 hours", "Last 7 days", "Last 30 days"],
            index=3,
            key="date_preset",
        )
        if preset == "Today":
            start_dt_utc = dt.combine(today_utc, dtime.min, tzinfo=timezone.utc)
            end_dt_utc   = dt.combine(today_utc, dtime.max, tzinfo=timezone.utc)
        elif preset == "Yesterday":
            y = today_utc - timedelta(days=1)
            start_dt_utc = dt.combine(y, dtime.min, tzinfo=timezone.utc)
            end_dt_utc   = dt.combine(y, dtime.max, tzinfo=timezone.utc)
        elif preset == "Last 24 hours":
            end_dt_utc   = dt.now(timezone.utc)
            start_dt_utc = end_dt_utc - timedelta(hours=24)
        elif preset == "Last 7 days":
            start_dt_utc = dt.combine(today_utc - timedelta(days=6), dtime.min, tzinfo=timezone.utc)
            end_dt_utc   = dt.combine(today_utc, dtime.max, tzinfo=timezone.utc)
        else:  # Last 30 days
            start_dt_utc = dt.combine(today_utc - timedelta(days=29), dtime.min, tzinfo=timezone.utc)
            end_dt_utc   = dt.combine(today_utc, dtime.max, tzinfo=timezone.utc)

        submitted = st.form_submit_button("🔎 Generate / Refresh", use_container_width=True)

st.title("📰 Sentiment Analysis – Mock POC")

# -------------------------------------
# Generate on submit (no rerun on mere widget changes)
# -------------------------------------
if submitted or st.session_state.get("news_df") is None:
    frames = []
    subject_parts = []
    subj_codes = []

    if mode in ("Company", "Both") and company_name:
        frames.append(
            generate_mock_news(
                subject=company_name, code=company_ticker, kind="company",
                n=N_ITEMS, start_dt_utc=start_dt_utc, end_dt_utc=end_dt_utc,
                mu=MU, sigma=SIGMA, seed_val=SEED
            )
        )
        subject_parts.append(f"{company_name} ({company_ticker})")
    if mode in ("Topic", "Both") and (topic_name or "").strip():
        frames.append(
            generate_mock_news(
                subject=topic_name, code=topic_code, kind="topic",
                n=N_ITEMS, start_dt_utc=start_dt_utc, end_dt_utc=end_dt_utc,
                mu=MU, sigma=SIGMA, seed_val=SEED
            )
        )
        subject_parts.append(f"Topic: {topic_name}")
    if mode in ("Topic", "Both") and not (topic_name or "").strip():
        st.warning("Please enter a topic.")

    if frames:
        news_df_all = pd.concat(frames, ignore_index=True).sort_values("datetime_utc", ascending=False)
        st.session_state["news_df"] = news_df_all
        st.session_state["subject_display"] = " + ".join(subject_parts) or "Selection"
        st.session_state["subject_code"] = "-".join([c for c in [company_ticker if mode in ("Company","Both") else None,
                                                                 topic_code if mode in ("Topic","Both") else None] if c]) or "SUBJ"
        st.session_state["generated_at"] = datetime.now(timezone.utc)

        # Auto-generate executive summary (no button)
        default_prompt = (
            "You are an equity research assistant. Read the news and produce a concise, executive-style brief "
            "for a busy trading desk. Summarise key positives, key risks, and any near-term catalysts (1–4 weeks). "
            "Finish with a 1-sentence bottom line."
        )
        st.session_state["summary_text"] = mock_llm_summary(st.session_state["news_df"], st.session_state["subject_display"], default_prompt)

news_df = st.session_state.get("news_df")
subject_display = st.session_state.get("subject_display")
subject_code = st.session_state.get("subject_code", "SUBJ")

# -------------------------------------
# Main view
# -------------------------------------
if news_df is not None and not news_df.empty:
    # Header KPIs (no filters; full dataset)
    as_of_ts = pd.to_datetime(news_df["datetime_utc"]).max()
    gen_at = st.session_state.get("generated_at")
    k1, k2, k3 = st.columns(3)
    k1.metric("Items", len(news_df))
    k2.metric("Sentiment as of", as_of_ts.strftime("%d %b %Y %H:%M UTC"))
    k3.metric("Generated at", gen_at.strftime("%d %b %Y %H:%M UTC") if gen_at else "—")

    # Charts (no filters, colour by scope if Both)
    scopes = sorted(news_df["scope"].unique())
    color_arg = "scope" if len(scopes) > 1 else None

    st.subheader("Sentiment Score Distribution")
    fig_hist = px.histogram(news_df, x="sentiment", nbins=30, marginal="rug", color=color_arg)
    fig_hist.update_layout(height=300, bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Sentiment Box Plot")
    fig_box = px.box(news_df, y="sentiment", points="all", color=color_arg)
    fig_box.update_layout(height=300)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Sentiment over Time")
    tmp = news_df.sort_values("datetime_utc").copy()
    tmp["ts"] = pd.to_datetime(tmp["datetime_utc"])
    fig_time = px.scatter(tmp, x="ts", y="sentiment", hover_data=["scope", "source", "headline"],
                          trendline="lowess", color=color_arg)
    fig_time.update_layout(height=320)
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("Source Breakdown")
    src_counts = news_df.groupby(["source", "label", "scope"], as_index=False).size() if len(scopes) > 1 \
        else news_df.groupby(["source", "label"], as_index=False).size()
    fig_src = px.bar(src_counts, x="source", y="size",
                     color="label", barmode="stack",
                     facet_col="scope" if len(scopes) > 1 else None)
    fig_src.update_layout(height=360, xaxis_title=None, yaxis_title="Items")
    st.plotly_chart(fig_src, use_container_width=True)

    # Table
    st.subheader("News Items")
    display_cols = ["datetime_local", "scope", "source", "headline", "sentiment", "label", "url"]
    st.dataframe(
        news_df.sort_values("datetime_utc", ascending=False)[display_cols]
               .rename(columns={"datetime_local": "datetime"}),
        use_container_width=True,
        hide_index=True,
    )

    # Downloads
    st.download_button(
        label="⬇️ Download CSV",
        data=news_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{subject_code}_mock_news.csv",
        mime="text/csv",
    )

    # Executive summary (auto-generated on dataset refresh)
    st.subheader(f"Executive Summary (LLM – mocked) — {subject_display or 'Selection'}")
    st.markdown(st.session_state.get("summary_text", ""))

else:
    st.info("Use the sidebar to choose **Company**, **Topic**, or **Both**, pick a **Preset** or **Custom date & time**, then click **Generate / Refresh**.")
