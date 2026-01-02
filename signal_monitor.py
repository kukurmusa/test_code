# app.py
# Streamlit dashboard querying kdb with auto-refresh support

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

import pykx as kx
from streamlit_autorefresh import st_autorefresh  # AUTO REFRESH


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Signals Process Monitor", layout="wide")

TZ = timezone.utc
NOW = datetime.now(tz=TZ)
RESAMPLE_FREQ = "5min"


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Environment")

environment = st.sidebar.selectbox(
    "Select environment",
    ["PROD", "UAT", "DEV"],
    index=0,
)

st.sidebar.divider()

# AUTO REFRESH
auto_refresh = st.sidebar.toggle("Auto refresh", value=False)

refresh_interval = st.sidebar.selectbox(
    "Refresh interval",
    options=[10, 30, 60, 120],
    index=1,
    help="Seconds",
)

if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="signals_autorefresh")

st.sidebar.caption("Coverage: **Today only**")


# -----------------------------
# kdb connection
# -----------------------------
def _get_kdb_config() -> dict:
    if "kdb" in st.secrets:
        return dict(
            host=st.secrets["kdb"]["host"],
            port=int(st.secrets["kdb"]["port"]),
            username=st.secrets["kdb"].get("username", ""),
            password=st.secrets["kdb"].get("password", ""),
        )
    return dict(
        host=os.getenv("KDB_HOST", "localhost"),
        port=int(os.getenv("KDB_PORT", "5000")),
        username=os.getenv("KDB_USER", ""),
        password=os.getenv("KDB_PASS", ""),
    )


@st.cache_resource(show_spinner=False)
def get_kdb_conn() -> kx.QConnection:
    cfg = _get_kdb_config()
    q = kx.QConnection(
        host=cfg["host"],
        port=cfg["port"],
        username=cfg["username"],
        password=cfg["password"],
    )
    q.sync("1b")  # ping
    return q


def _today_filter(ts_col: str) -> str:
    return f"date {ts_col} = .z.d"


# -----------------------------
# kdb queries (ALL envs)
# -----------------------------
@st.cache_data(ttl=30, show_spinner=True)  # AUTO REFRESH respects TTL
def load_kdb_today_all_envs() -> dict[str, pd.DataFrame]:
    q = get_kdb_conn()

    df_durations = q.sync(f"""
        select ts, env, function, duration_ms, run_id, host, status
        from signal_durations
        where {_today_filter("ts")}
    """).pd()

    df_latency = q.sync(f"""
        select ts, env, path, function, latency_ms, run_id, host
        from signal_latency
        where {_today_filter("ts")}
    """).pd()

    df_errors = q.sync(f"""
        select ts, env, function, stage, message, host, run_id, stacktrace
        from signal_errors
        where {_today_filter("ts")}
    """).pd()

    df_runs = q.sync(f"""
        select start_ts, end_ts, env, run_id, status, total_duration_ms, error_count, host
        from signal_runs
        where {_today_filter("start_ts")}
    """).pd()

    # Ensure datetime columns
    for df, col in [
        (df_durations, "ts"),
        (df_latency, "ts"),
        (df_errors, "ts"),
        (df_runs, "start_ts"),
        (df_runs, "end_ts"),
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return dict(
        durations=df_durations,
        latency=df_latency,
        errors=df_errors.sort_values("ts", ascending=False),
        runs=df_runs,
    )


def q_by_time(df: pd.DataFrame, ts_col: str, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts", "p50", "p95"])
    return (
        df.set_index(ts_col)[value_col]
        .resample(RESAMPLE_FREQ)
        .quantile([0.5, 0.95])
        .unstack()
        .rename(columns={0.5: "p50", 0.95: "p95"})
        .reset_index()
    )


# -----------------------------
# Load data
# -----------------------------
try:
    data = load_kdb_today_all_envs()
except Exception as e:
    st.error("Failed to load data from kdb")
    st.code(str(e))
    st.stop()

df_durations = data["durations"].query("env == @environment")
df_latency = data["latency"].query("env == @environment")
df_errors = data["errors"].query("env == @environment")
df_runs = data["runs"].query("env == @environment")


# -----------------------------
# Header
# -----------------------------
st.title("Signals Process Monitor")
st.caption(
    f"Environment: **{environment}** • "
    f"Last updated: {datetime.now(tz=TZ):%Y-%m-%d %H:%M:%S} UTC"
)

# -----------------------------
# KPIs
# -----------------------------
runs_today = len(df_runs)
errors_today = len(df_errors)
success_rate = 100 * df_runs["status"].eq("success").mean() if runs_today else 0

p95_dur = df_durations["duration_ms"].quantile(0.95) if not df_durations.empty else np.nan
p95_lat = df_latency["latency_ms"].quantile(0.95) if not df_latency.empty else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Runs today", f"{runs_today:,}")
c2.metric("Success rate", f"{success_rate:.1f}%")
c3.metric("Errors today", f"{errors_today}")
c4.metric("P95 function duration (ms)", f"{p95_dur:.0f}" if np.isfinite(p95_dur) else "—")
c5.metric("P95 latency (ms)", f"{p95_lat:.0f}" if np.isfinite(p95_lat) else "—")

st.divider()


# -----------------------------
# Charts
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Function Duration")
    fn = st.selectbox("Function", sorted(df_durations["function"].unique()))
    chart = q_by_time(
        df_durations[df_durations["function"] == fn],
        "ts",
        "duration_ms",
    )
    st.line_chart(chart.set_index("ts")[["p50", "p95"]], height=300)

with right:
    st.subheader("Latency Between Source Tables")
    path = st.selectbox("Latency path", sorted(df_latency["path"].unique()))
    chart = q_by_time(
        df_latency[df_latency["path"] == path],
        "ts",
        "latency_ms",
    )
    st.line_chart(chart.set_index("ts")[["p50", "p95"]], height=220)

    st.subheader("Errors Logged Today")
    if df_errors.empty:
        st.success("No errors logged today.")
    else:
        st.dataframe(
            df_errors[["ts", "function", "stage", "message", "host", "run_id"]],
            use_container_width=True,
            height=220,
        )
        with st.expander("Most recent stack trace"):
            st.code(df_errors.iloc[0]["stacktrace"], language="text")

st.divider()


# -----------------------------
# Runs
# -----------------------------
st.subheader("Recent Runs")
st.dataframe(
    df_runs.sort_values("start_ts", ascending=False).head(50),
    use_container_width=True,
    height=360,
)
