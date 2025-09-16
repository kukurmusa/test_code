# app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# --- External helpers (implement in services.py) -----------------------------
# get_news_data(type_of_search: str, search_string: str, date_preset: str) -> {"res": DataFrame, "info": {...}}
# generate_response(client_or_token, text: str, system_prompt: str) -> {"content": "..."}
# extract_sentiment_score(text: str) -> Optional[float]
# get_oauth_token() -> str | client
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
MAX_ROWS_FOR_LLM = 120

# ---- Price/Predictions config (yours, kept verbatim where it matters) -------
PRICE_CFG = {
    "use_sample": True,          # Set False to show file uploaders (or replace with real API)
    "default_horizon": 5,        # e.g., 5/10/15
    "show_class_markers": True,
    "show_up_init": True,
    "show_down_init": True,
    "show_flat_init": False,
    "marker_size": 9,
    "marker_opacity": 0.85,
    "bps_bar_width_min": 2,
    "show_volume": True,
    "vol_axis_range_multiplier": 4.0,
    "vol_opacity": 0.35,
    "vol_bar_width_min": 1,
}

# --- State & logging ---------------------------------------------------------
def init_state() -> None:
    # Scope defaults
    st.session_state.setdefault("mode", "Company")                        # "Company" | "Topic" | "Company + Topic"
    st.session_state.setdefault("company_input_mode", "Pick from list")   # "Pick from list" | "Type manually"
    st.session_state.setdefault("company_name", COMPANIES[2]["name"])     # default Alphabet
    st.session_state.setdefault("topic_name", "")
    st.session_state.setdefault("date_preset", "this_week")

    # Filters defaults
    st.session_state.setdefault("selected_sources", None)                 # list[str] or None
    st.session_state.setdefault("sentiment_bucket", "All")                # "All" | "Positive" | "Neutral" | "Negative"
    st.session_state.setdefault("keyword", "")

    # Polling/data
    st.session_state.setdefault("polling", False)
    st.session_state.setdefault("news", None)

    # New-content detection (robust to rolling windows)
    st.session_state.setdefault("_latest_ts", None)        # most recent article timestamp (tz-aware)
    st.session_state.setdefault("_latest_ts_count", 0)     # number of rows sharing that timestamp

    # Summary state
    st.session_state.setdefault("summary_stale", True)     # set True when new articles or scope changes
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("llm_sentiment", None)

    # UI timestamps
    st.session_state.setdefault("last_update", None)       # shows in Status Panel

    # Logger
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

# --- Data fetch & polling ----------------------------------------------------
def fetch_and_update_news() -> None:
    """Fetch, normalise, update state; mark summary_stale only on truly newer content."""
    mode = st.session_state.mode
    preset_key = st.session_state.date_preset
    preset_val = DATE_PRESETS[preset_key]

    if mode == "Company":
        subject = (st.session_state.company_name or "").strip()
        res = get_news_data("COMPANY", subject, preset_val)
        subtitle = f"Company – {subject}"

    elif mode == "Topic":
        subject = (st.session_state.topic_name or "").strip()
        res = get_news_data("TOPIC", subject, preset_val)
        subtitle = f"Topic – {subject}"

    elif mode == "Company + Topic":
        company = (st.session_state.company_name or "").strip()
        topic = (st.session_state.topic_name or "").strip()
        subject = f"{company}|{topic}"
        res = get_news_data("COMPANY_TOPIC", subject, preset_val)
        subtitle = f"Company + Topic – {company} / {topic}"

    else:
        res = {"res": pd.DataFrame(), "info": {}}
        subtitle = "Invalid mode"

    df = _normalise_df(res.get("res") if isinstance(res, dict) else pd.DataFrame())

    # New-content detection via latest ts (+ tie count)
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
    """On every run, if polling is ON, fetch & note changes."""
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

# --- Visuals -----------------------------------------------------------------
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

# Compact status bar with toggle
def status_panel():
    st.markdown("#### 🟢 Status Panel")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    polling = "✅ ON" if st.session_state.polling else "⛔ OFF"
    last_refresh = (st.session_state.last_update.strftime("%H:%M:%S UTC")
                    if st.session_state.last_update else "—")
    latest_article = (st.session_state["_latest_ts"].strftime("%Y-%m-%d %H:%M:%S")
                      if st.session_state["_latest_ts"] else "—")

    col1.caption(f"**Polling:** {polling}")
    col2.caption(f"**Last refresh:** {last_refresh}")
    col3.caption(f"**Latest article:** {latest_article}")

    if col4.button("🔁 Toggle", use_container_width=True):
        st.session_state.polling = not st.session_state.polling
        st.toast("Polling started" if st.session_state.polling else "Polling stopped")
        log_event("poll_toggle", {"on": st.session_state.polling})
        st.experimental_rerun()

# Top words chart
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

# === PRICE / PREDICTIONS helpers (from your snippet, modularised) =============
def make_sample_price(start=None, minutes=390, seed=42):
    rng = np.random.default_rng(seed)
    if start is None:
        start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    ts = pd.date_range(start=start, periods=minutes, freq="1min")
    rets = rng.normal(0, 0.0008, size=minutes)
    price = 100 * (1 + rets).cumprod()
    base = rng.lognormal(mean=9.2, sigma=0.5, size=minutes)
    minute = np.arange(minutes)
    u_shape = 0.7 + 0.6 * (np.cos((minute / minutes) * 2 * np.pi) * -1)
    vol = (base * u_shape).astype(int)
    return pd.DataFrame({"timestamp": ts, "price": price, "volume": vol})

def make_sample_preds(price_df: pd.DataFrame, horizons=(5,10,15), seed=43):
    rng = np.random.default_rng(seed)
    rows = []
    for t in price_df["timestamp"]:
        for h in horizons:
            cls = rng.choice(["Up", "Down", "Flat"], p=[0.35, 0.35, 0.30])
            conf = rng.uniform(0.55, 0.95)
            pred_bps = rng.normal(0, 6)
            rows.append({"timestamp": t, "horizon_min": h, "class": cls, "confidence": conf, "pred_bps": pred_bps})
    return pd.DataFrame(rows)

def ensure_ts(df: pd.DataFrame, col="timestamp") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])

def build_price_pred_fig(df_price: pd.DataFrame, df_pred: pd.DataFrame, cfg: Dict[str, Any]) -> go.Figure:
    df_price = ensure_ts(df_price, "timestamp").sort_values("timestamp")
    for col in ["price", "volume"]:
        if col not in df_price.columns:
            raise ValueError("Price data must include 'price' and 'volume' columns (1-minute binned).")

    df_pred = ensure_ts(df_pred, "timestamp").sort_values("timestamp")
    needed = {"horizon_min", "class", "pred_bps"}
    missing = needed - set(df_pred.columns)
    if missing:
        raise ValueError(f"Predictions data missing columns: {missing}. Required: timestamp, horizon_min, class, pred_bps[, confidence].")

    # Compute END time & realised END price
    df_pred["end_time"] = df_pred["timestamp"] + pd.to_timedelta(df_pred["horizon_min"], unit="m")
    df_price_end = df_price.rename(columns={"timestamp": "end_time", "price": "end_price"})
    df_pred = df_pred.merge(df_price_end[["end_time", "end_price"]], on="end_time", how="left").dropna(subset=["end_price"])

    # Horizon picker
    available_horizons = sorted(df_pred["horizon_min"].dropna().unique().tolist())
    default_idx = available_horizons.index(cfg["default_horizon"]) if cfg["default_horizon"] in available_horizons else 0
    selected_h = st.selectbox("Prediction horizon (minutes):", available_horizons, index=default_idx, key="horizon_select")

    dfh = df_pred[df_pred["horizon_min"] == selected_h].copy()
    dfh["class"] = dfh["class"].astype(str).str.capitalize()

    # Figure scaffold
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{}]]
    )

    # Top: Price line
    fig.add_trace(
        go.Scatter(
            x=df_price["timestamp"], y=df_price["price"],
            mode="lines", name="Trade Price",
            hovertemplate="Time: %{x}<br>Price: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1, secondary_y=False
    )

    # Top: Volume (secondary axis)
    if cfg["show_volume"]:
        vol95 = float(np.nanpercentile(df_price["volume"], 95))
        vol_axis_max = vol95 * cfg["vol_axis_range_multiplier"]
        fig.add_trace(
            go.Bar(
                x=df_price["timestamp"], y=df_price["volume"],
                name="Volume (1m)",
                width=int(cfg["vol_bar_width_min"] * 60 * 1000),
                opacity=cfg["vol_opacity"], marker_line_width=0,
                hovertemplate="Time: %{x}<br>Volume: %{y:,}<extra></extra>",
            ),
            row=1, col=1, secondary_y=True
        )
        fig.update_yaxes(
            title_text="", showgrid=False, showticklabels=False, zeroline=False,
            range=[0, max(vol_axis_max, df_price["volume"].max() * 1.05)],
            secondary_y=True, row=1, col=1
        )

    # Top: Classification markers
    if cfg["show_class_markers"] and not dfh.empty:
        class_to_symbol = {"Up": "triangle-up", "Down": "triangle-down", "Flat": "circle"}
        class_init_vis = {"Up": cfg["show_up_init"], "Down": cfg["show_down_init"], "Flat": cfg["show_flat_init"]}
        for cls in ["Up", "Down", "Flat"]:
            dfg = dfh[dfh["class"] == cls]
            if dfg.empty:
                continue
            init_visible = True if class_init_vis.get(cls, True) else "legendonly"
            if "confidence" in dfg.columns:
                custom = np.stack([dfg["timestamp"].dt.strftime("%Y-%m-%d %H:%M"), dfg["confidence"]], axis=1)
                hover = (
                    "END Time: %{x}<br>END Price: %{y:.4f}"
                    f"<br>Pred Time: %{{customdata[0]}}<br>Horizon: {selected_h}m<br>Class: {cls}"
                    "<br>Confidence: %{{customdata[1]:.0%}}<extra></extra>"
                )
            else:
                custom = np.stack([dfg["timestamp"].dt.strftime("%Y-%m-%d %H:%M")], axis=1)
                hover = (
                    "END Time: %{x}<br>END Price: %{y:.4f}"
                    f"<br>Pred Time: %{{customdata[0]}}<br>Horizon: {selected_h}m<br>Class: {cls}<extra></extra>"
                )

            fig.add_trace(
                go.Scatter(
                    x=dfg["end_time"], y=dfg["end_price"], mode="markers",
                    name=f"{selected_h}m {cls}",
                    marker=dict(size=cfg["marker_size"], symbol=class_to_symbol[cls]),
                    opacity=cfg["marker_opacity"], hovertemplate=hover, customdata=custom,
                    visible=init_visible, legendgroup=f"{selected_h}m {cls}",
                ),
                row=1, col=1, secondary_y=False
            )

    # Bottom: Δbps bars
    if not dfh.empty:
        colours = np.where(dfh["pred_bps"] > 0, "green", np.where(dfh["pred_bps"] < 0, "red", "grey"))
        fig.add_trace(
            go.Bar(
                x=dfh["end_time"], y=dfh["pred_bps"],
                name=f"{selected_h}m Δbps (pred)",
                width=int(cfg["bps_bar_width_min"] * 60 * 1000),
                marker_color=colours, opacity=0.9,
                hovertemplate=("END Time: %{x}<br>Predicted Δ: %{y:.2f} bps"
                               f"<br>Horizon: {selected_h}m"
                               + ("<br>Confidence: %{{customdata[0]:.0%}}" if "confidence" in dfh.columns else "")
                               + "<extra></extra>"),
                customdata=dfh[["confidence"]] if "confidence" in dfh.columns else None,
            ),
            row=2, col=1
        )

    # Layout polish
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Predicted Δ (bps)", row=2, col=1, zeroline=True, zerolinewidth=1)
    fig.update_layout(
        height=840, margin=dict(l=10, r=10, t=32, b=10),
        hovermode="x unified", barmode="overlay",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=False)),
    )
    fig.update_xaxes(matches='x', row=1, col=1, showspikes=True, spikemode="across")
    fig.update_xaxes(matches='x', row=2, col=1, showspikes=True, spikemode="across")
    return fig

def load_price_and_preds(company_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace this with your real data hookups when ready.
    For now:
      - if PRICE_CFG['use_sample'] is True -> generate synthetic intraday + preds
      - else -> show uploaders in the sidebar to load CSVs
    Expected schemas:
      Price: timestamp, price, volume   (1-minute rows)
      Preds: timestamp(pred time), horizon_min, class, pred_bps[, confidence]
    """
    if PRICE_CFG["use_sample"]:
        df_price = make_sample_price()
        df_pred = make_sample_preds(df_price)
        return df_price, df_pred
    else:
        st.info("Upload Price & Predictions CSVs for the selected company.")
        price_file = st.sidebar.file_uploader("Price CSV (timestamp, price, volume)", type=["csv"], key="price_csv")
        preds_file = st.sidebar.file_uploader("Predictions CSV (timestamp, horizon_min, class, pred_bps[, confidence])", type=["csv"], key="preds_csv")
        if not price_file or not preds_file:
            st.warning("Upload both files to render the chart.")
            return pd.DataFrame(), pd.DataFrame()
        return pd.read_csv(price_file), pd.read_csv(preds_file)

# --- App ---------------------------------------------------------------------
def main() -> None:
    init_state()

    # True polling only when ON
    if st.session_state.polling:
        st_autorefresh(interval=POLL_INTERVAL * 1000, key="poller")

    st.title("Sentiment Analysis")
    st.divider()

    maybe_poll()

    news = st.session_state.news or {}
    df_all = news.get("docs", pd.DataFrame())
    subtitle = news.get("info", {}).get("subtitle", "")

    status_panel()
    st.divider()

    # Sidebar: Scope + Filters (key-based)
    with st.sidebar:
        st.header("Scope")

        old_scope = {
            "mode": st.session_state.mode,
            "company_input_mode": st.session_state.company_input_mode,
            "company_name": st.session_state.company_name,
            "topic_name": st.session_state.topic_name,
            "date_preset": st.session_state.date_preset,
        }

        st.radio("Analyse by", ["Company", "Topic", "Company + Topic"], key="mode")

        preset_keys = list(DATE_PRESETS.keys())
        st.selectbox(
            "Date range (UTC)", preset_keys,
            index=preset_keys.index(st.session_state.date_preset),
            key="date_preset",
        )

        if st.session_state.mode in ["Company", "Company + Topic"]:
            st.radio("Company input", ["Pick from list", "Type manually"], key="company_input_mode")
            if st.session_state.company_input_mode == "Pick from list":
                company_names = [c["name"] for c in COMPANIES]
                idx = company_names.index(st.session_state.company_name) if st.session_state.company_name in company_names else 0
                st.selectbox("Company", company_names, index=idx, key="company_name")
            else:
                st.text_input("Company (name or ticker)", key="company_name")

        if st.session_state.mode in ["Topic", "Company + Topic"]:
            st.text_input("Topic", key="topic_name")

        if st.button("🧠 Get Sentiment Analysis", use_container_width=True):
            fetch_and_update_news()
            st.experimental_rerun()

        if any(st.session_state[k] != old_scope[k] for k in old_scope.keys()):
            st.session_state.summary_stale = True

        st.markdown("---")
        st.subheader("Filters")

        if "source_name" in df_all.columns:
            all_sources = sorted(df_all["source_name"].dropna().unique())
            if st.session_state.selected_sources is None:
                st.session_state.selected_sources = all_sources[:]  # default = all
            st.multiselect("Source", options=all_sources,
                           default=st.session_state.selected_sources, key="selected_sources")
        else:
            st.session_state.selected_sources = None

        st.selectbox("Sentiment bucket", ["All", "Positive", "Neutral", "Negative"], key="sentiment_bucket")
        st.text_input("Keyword search in headlines/snippets", key="keyword")

    # Apply filters
    df = df_all.copy()
    sel_sources = st.session_state.selected_sources
    if sel_sources:
        df = df[df["source_name"].isin(sel_sources)]
    bucket = st.session_state.sentiment_bucket
    if bucket != "All" and "sentiment" in df.columns:
        s = pd.to_numeric(df["sentiment"], errors="coerce")
        if bucket == "Positive":
            df = df[s > 0.1]
        elif bucket == "Negative":
            df = df[s < -0.1]
        else:
            df = df[s.between(-0.1, 0.1)]
    kw = st.session_state.keyword
    if kw:
        mask = pd.Series(False, index=df.index)
        for col in [c for c in df.columns if any(k in c.lower() for k in ["title", "summary", "content"])]:
            mask |= df[col].astype(str).str.contains(kw, case=False, na=False)
        df = df[mask]

    # Results
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

        if "sentiment" in df.columns:
            styled_df = (
                df.style
                .format({"sentiment": "{:.2f}"})
                .set_properties(subset=["sentiment"], **{"font-weight": "bold"})
                .background_gradient(subset=["sentiment"], cmap="RdYlGn", vmin=-1, vmax=1)
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True, column_config=cfg)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True, column_config=cfg)

        fig_words = plot_top_words(df, ["title", "summary"])
        if fig_words:
            st.plotly_chart(fig_words, use_container_width=True)
    else:
        st.info("No news yet. Choose a scope and click **Get Sentiment Analysis**.")

    # --- Executive summary (filtered df) -------------------------------------
    if not df.empty:
        st.subheader("Executive Summary")
        refresh_summary = st.button("🔄 Refresh Executive Summary", use_container_width=True)

        if st.session_state.summary_stale or refresh_summary:
            prompt = (
                "Summarise the following news items.\n"
                "End with one line: **Sentiment Score:** <NUMBER between -1 and 1>\n"
            )
            client = get_oauth_token()
            df_for_llm = df.sort_values("timestamp", ascending=False).head(MAX_ROWS_FOR_LLM) if "timestamp" in df.columns else df.head(MAX_ROWS_FOR_LLM)
            with st.spinner("Summarising..."):
                resp = generate_response(client, df_for_llm.to_json(orient="records"), prompt)
                content = resp.get("content", "")
                score = extract_sentiment_score(content)
                st.session_state.summary_text = content
                st.session_state.llm_sentiment = float(score) if score is not None else None
                st.session_state.summary_stale = False
                st.session_state.last_update = datetime.utcnow()
            log_event("summary_generated", {
                "latest_ts": st.session_state["_latest_ts"].isoformat() if st.session_state["_latest_ts"] else None,
                "filtered_rows": int(len(df_for_llm)),
            })

        if st.session_state.llm_sentiment is not None:
            st.metric("LLM Sentiment", f"{st.session_state.llm_sentiment:.2f}")
        st.markdown(st.session_state.summary_text)

    # === ───────────────────────── Price / Volume / Predictions section ─────────────────────────
    if st.session_state.mode in ["Company", "Company + Topic"]:
        company_for_prices = st.session_state.company_name or "Selected Company"
        with st.expander(f"📈 Price, Volume & Predicted Δbps — {company_for_prices}", expanded=False):
            # In future: replace load_price_and_preds() with real API calls keyed by ticker/company
            df_price, df_pred = load_price_and_preds(company_for_prices)
            if not df_price.empty and not df_pred.empty:
                try:
                    fig_p = build_price_pred_fig(df_price, df_pred, PRICE_CFG)
                    st.plotly_chart(fig_p, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render price/predictions chart: {e}")
            else:
                st.info("Provide price & predictions data (using sample mode or CSV uploaders) to show this chart.")
            # Optional: small notes / schema reminder
            with st.expander("Data format & behaviour", expanded=False):
                st.markdown("""
**Price CSV**: `timestamp`, `price`, `volume` (1-minute binned)  
**Predictions CSV**: `timestamp` (prediction time), `horizon_min` (int), `class` (Up/Down/Flat), `pred_bps` (float), optional `confidence`.

- Volume is drawn on the **secondary Y-axis** of the **top chart** as subtle bars.  
- Use the legend to show/hide **Up/Down/Flat** markers (Flat hidden by default).  
- Bottom panel shows **predicted Δ (bps)** for selected horizon.  
""")

    st.caption("© MAAS Execution Analytics")


if __name__ == "__main__":
    main()
