# app.py
# Streamlit Sentiment Analysis POC (mock data)
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
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

# Fixed mock controls (kept out of the UI)
N_ITEMS = 60                # total items generated per “subject”
MU = 0.05
SIGMA = 0.35
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------
# Helpers
# -------------------------------------
def stable_int_from_string(s: str) -> int:
    """Deterministic, cross-session integer from a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

@st.cache_data(show_spinner=False)
def generate_mock_news(subject: str, code: str, kind: str,
                       n: int, start_dt_utc: datetime, end_dt_utc: datetime,
                       mu: float, sigma: float, seed_val: int) -> pd.DataFrame:
    """Generate mock articles for a company/topic between start/end (UTC)."""
    # Stable RNG per subject
    subj_seed = seed_val + stable_int_from_string(f"{kind}:{code}") % 10_000_000
    rng = np.random.default_rng(subj_seed)

    # Random timestamps uniformly within the window
    span_seconds = (end_dt_utc - start_dt_utc).total_seconds()
    rand_secs = rng.random(n) * span_seconds
    times = [start_dt_utc + timedelta(seconds=float(s)) for s in rand_secs]
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
            "datetime_utc": times[i].replace(microsecond=0),
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
    # Convenience local-time column for display (naive)
    df["datetime_local"] = pd.to_datetime(df["datetime_utc"]).dt.tz_convert(None)
    return df

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
# Sidebar – Scope + Date range
# -------------------------------------
with st.sidebar:
    st.header("🔎 Scope")

    mode = st.radio("Analyse by", ["Company", "Topic", "Both"], horizontal=True)

    # Defaults
    subject_company_name = None
    subject_company_code = None
    subject_topic_name = None
    subject_topic_code = None

    if mode in ("Company", "Both"):
        options = [f"{c['name']} ({c['ticker']})" for c in COMPANIES]
        choice = st.selectbox("Company", options, index=0)
        idx = options.index(choice)
        subject_company_name = COMPANIES[idx]["name"]
        subject_company_code = COMPANIES[idx]["ticker"]

    if mode in ("Topic", "Both"):
        subject_topic_name = st.text_input("Topic", placeholder="e.g., AI in data centres")
        subject_topic_code = (subject_topic_name or "TOPIC").strip()

    st.markdown("---")
    # Date range picker (UTC inclusive). Default = last 7 days.
    today_utc = datetime.now(timezone.utc).date()
    default_start = today_utc - timedelta(days=6)
    start_date, end_date = st.date_input(
        "Date range (UTC)",
        value=(default_start, today_utc),
        min_value=today_utc - timedelta(days=365),
        max_value=today_utc
    )

    # Safety: ensure tuple even if single date returned
    if isinstance(start_date, date) and isinstance(end_date, date):
        start_dt_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt_utc = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    else:
        # Fallback to last 7 days
        start_dt_utc = datetime.combine(default_start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt_utc = datetime.combine(today_utc, datetime.max.time(), tzinfo=timezone.utc)

    st.markdown("---")
    generate = st.button("🔎 Generate Mock News", use_container_width=True)

st.title("📰 Sentiment Analysis – Mock POC")

# -------------------------------------
# Generate upon click
# -------------------------------------
if generate:
    frames = []
    subject_display_parts = []

    if mode in ("Company", "Both") and subject_company_name:
        frames.append(
            generate_mock_news(
                subject=subject_company_name,
                code=subject_company_code,
                kind="company",
                n=N_ITEMS,
                start_dt_utc=start_dt_utc,
                end_dt_utc=end_dt_utc,
                mu=MU, sigma=SIGMA, seed_val=SEED
            )
        )
        subject_display_parts.append(f"{subject_company_name} ({subject_company_code})")

    if mode in ("Topic", "Both") and (subject_topic_name or "").strip():
        frames.append(
            generate_mock_news(
                subject=subject_topic_name,
                code=subject_topic_code,
                kind="topic",
                n=N_ITEMS,
                start_dt_utc=start_dt_utc,
                end_dt_utc=end_dt_utc,
                mu=MU, sigma=SIGMA, seed_val=SEED
            )
        )
        subject_display_parts.append(f"Topic: {subject_topic_name}")
    elif mode in ("Topic", "Both") and not (subject_topic_name or "").strip():
        st.warning("Please enter a topic to generate mock news.")

    if frames:
        news_df_all = pd.concat(frames, ignore_index=True).sort_values("datetime_utc", ascending=False)
        st.session_state["news_df"] = news_df_all
        st.session_state["subject_display"] = " + ".join(subject_display_parts)
        st.session_state["subject_code"] = "-".join(
            [f for f in [subject_company_code, subject_topic_code] if f]
        ) or "SUBJ"
    else:
        st.session_state["news_df"] = pd.DataFrame()
        st.session_state["subject_display"] = None

news_df = st.session_state.get("news_df")
subject_display = st.session_state.get("subject_display") or ""
subject_code = st.session_state.get("subject_code", "SUBJ")

# -------------------------------------
# Main view
# -------------------------------------
if news_df is not None and not news_df.empty:
    # Filters
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        # Scope filter only appears when multiple scopes exist
        scopes = sorted(news_df["scope"].unique())
        if len(scopes) > 1:
            scopes_sel = c1.multiselect("Scope", scopes, default=scopes)
        else:
            scopes_sel = scopes

        labels_sel = c2.multiselect("Labels", ["Positive", "Neutral", "Negative"],
                                    default=["Positive", "Neutral", "Negative"])
        sources_sel = c3.multiselect("Sources", sorted(news_df["source"].unique()),
                                     default=list(sorted(news_df["source"].unique())))
        min_s, max_s = c4.slider("Score range", -1.0, 1.0, (-1.0, 1.0), 0.05)

    mask = news_df["label"].isin(labels_sel) & news_df["source"].isin(sources_sel) & news_df["sentiment"].between(min_s, max_s)
    if len(scopes) > 1:
        mask &= news_df["scope"].isin(scopes_sel)

    view_df = news_df.loc[mask].copy()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Items", len(view_df))
    col2.metric("Avg sentiment", f"{view_df['sentiment'].mean():.2f}" if len(view_df) else "—")
    col3.metric("% Positive", f"{(view_df['label'].eq('Positive').mean()*100):.0f}%" if len(view_df) else "—")
    col4.metric("% Negative", f"{(view_df['label'].eq('Negative').mean()*100):.0f}%" if len(view_df) else "—")

    # Charts
    st.subheader("Sentiment Score Distribution")
    fig_hist = px.histogram(view_df, x="sentiment", nbins=30, marginal="rug", color="scope" if len(scopes) > 1 else None)
    fig_hist.update_layout(height=300, bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Sentiment Box Plot")
    fig_box = px.box(view_df, y="sentiment", points="all", color="scope" if len(scopes) > 1 else None)
    fig_box.update_layout(height=300)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Sentiment over Time")
    tmp = view_df.sort_values("datetime_utc").copy()
    tmp["ts"] = pd.to_datetime(tmp["datetime_utc"])
    fig_time = px.scatter(tmp, x="ts", y="sentiment", hover_data=["scope", "source", "headline"],
                          trendline="lowess", color="scope" if len(scopes) > 1 else None)
    fig_time.update_layout(height=320)
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("Source Breakdown")
    src_counts = view_df.groupby(["source", "label", "scope"], as_index=False).size() if len(scopes) > 1 \
        else view_df.groupby(["source", "label"], as_index=False).size()
    fig_src = px.bar(src_counts, x="source", y="size",
                     color="label", barmode="stack",
                     facet_col="scope" if len(scopes) > 1 else None)
    fig_src.update_layout(height=360, xaxis_title=None, yaxis_title="Items")
    st.plotly_chart(fig_src, use_container_width=True)

    # Table
    st.subheader("News Items")
    display_cols = ["datetime_local", "scope", "source", "headline", "sentiment", "label", "url"]
    st.dataframe(
        view_df.sort_values("datetime_utc", ascending=False)[display_cols]
               .rename(columns={"datetime_local": "datetime"}),
        use_container_width=True,
        hide_index=True,
    )

    # Downloads
    st.download_button(
        label="⬇️ Download CSV (all rows)",
        data=news_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{subject_code}_mock_news.csv",
        mime="text/csv",
    )
    st.download_button(
        label="⬇️ Download CSV (filtered)",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{subject_code}_mock_news_filtered.csv",
        mime="text/csv",
    )

    # Executive summary – dynamic prompt
    st.subheader(f"Executive Summary (LLM – mocked) — {subject_display or 'Selection'}")
    default_prompt = (
        "You are an equity research assistant. Read the news and produce a concise, executive-style brief "
        "for a busy trading desk. Summarise key positives, key risks, and any near-term catalysts (1–4 weeks). "
        "Finish with a 1-sentence bottom line."
    )
    with st.expander("Prompt", expanded=True):
        user_prompt = st.text_area("Edit the prompt", value=default_prompt, height=140)

    if st.button("🧠 Generate Executive Summary (mock)", use_container_width=True):
        st.session_state["summary_text"] = mock_llm_summary(view_df if len(view_df) else news_df,
                                                            subject_display or "Selection",
                                                            user_prompt)

    if st.session_state.get("summary_text"):
        st.markdown(st.session_state["summary_text"])
        st.download_button(
            label="⬇️ Download summary (Markdown)",
            data=st.session_state["summary_text"].encode("utf-8"),
            file_name=f"{subject_code}_summary.md",
            mime="text/markdown",
        )
else:
    st.info("Choose **Company**, **Topic**, or **Both** in the left sidebar, set a **Date range**, then click **Generate Mock News**.")
