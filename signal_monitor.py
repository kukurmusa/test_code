# app.py
# Streamlit dashboard (today-only) that QUERIES kdb+.
# Assumes you have PyKX installed and network access to your kdb process.
#
# Expected kdb tables (edit names/columns in the Q queries below to match yours):
#   - signal_durations : ts, env, function, duration_ms, run_id, host, status
#   - signal_latency   : ts, env, path, function, latency_ms, run_id, host
#   - signal_errors    : ts, env, function, stage, message, host, run_id, severity, error_type, stacktrace
#   - signal_runs      : start_ts, end_ts, env, run_id, status, total_duration_ms, error_count, host
#
# Install:
#   pip install streamlit pandas numpy pykx
#
# Run:
#   streamlit run app.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

import pykx as kx


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Signals Process Monitor", layout="wide")

TZ = timezone.utc
NOW = datetime.now(tz=TZ)
RESAMPLE_FREQ = "5min"  # fixed granularity


# -----------------------------
# kdb connection + queries
# -----------------------------
def _get_kdb_config() -> dict:
    """
    Reads config from:
      1) Streamlit secrets (recommended)
      2) Environment variables
    """
    # Streamlit secrets: .streamlit/secrets.toml
    # [kdb]
    # host="..."
    # port=...
    # username="..."
    # password="..."
    cfg = {}
    if "kdb" in st.secrets:
        cfg["host"] = st.secrets["kdb"].get("host")
        cfg["port"] = int(st.secrets["kdb"].get("port"))
        cfg["username"] = st.secrets["kdb"].get("username", "")
        cfg["password"] = st.secrets["kdb"].get("password", "")
    else:
        cfg["host"] = os.getenv("KDB_HOST", "localhost")
        cfg["port"] = int(os.getenv("KDB_PORT", "5000"))
        cfg["username"] = os.getenv("KDB_USER", "")
        cfg["password"] = os.getenv("KDB_PASS", "")

    return cfg


@st.cache_resource(show_spinner=False)
def get_kdb_conn() -> kx.QConnection:
    cfg = _get_kdb_config()
    # If your kdb is unauthenticated, username/password can be empty strings.
    conn = kx.QConnection(
        host=cfg["host"],
        port=cfg["port"],
        username=cfg.get("username", ""),
        password=cfg.get("password", ""),
    )
    # Quick sanity ping (optional)
    conn.sync("1b")
    return conn


def _q_today_filter(ts_col: str) -> str:
    """
    Builds a kdb expression that filters 'today' based on a timestamp column.
    Using .z.d for current date, and casting timestamp to date.
    """
    # date ts = .z.d
    return f"date {ts_col} = .z.d"


@st.cache_data(ttl=30, show_spinner=True)
def load_kdb_all_envs() -> dict[str, pd.DataFrame]:
    """
    Loads ALL envs for TODAY from kdb, one query per dataset (durations, latency, errors, runs),
    then Streamlit filters in-memory by selected env.
    """
    q = get_kdb_conn()

    # --- EDIT THESE TABLE NAMES / COLUMN NAMES AS NEEDED ---
    q_durations = f"""
        select ts, env, function, duration_ms, run_id, host, status
        from signal_durations
        where {_q_today_filter("ts")}
    """

    q_latency = f"""
        select ts, env, path, function, latency_ms, run_id, host
        from signal_latency
        where {_q_today_filter("ts")}
    """

    q_errors = f"""
        select ts, env, severity, function, stage, message, error_type, host, run_id, stacktrace
        from signal_errors
        where {_q_today_filter("ts")}
    """

    q_runs = f"""
        select start_ts, end_ts, env, run_id, status, total_duration_ms, error_count, host
        from signal_runs
        where {_q_today_filter("start_ts")}
    """

    try:
        df_durations = q.sync(q_durations).pd()
        df_latency = q.sync(q_latency).pd()
        df_errors = q.sync(q_errors).pd()
        df_runs = q.sync(q_runs).pd()
    except Exception as e:
        # Show a clean error and re-raise
        raise RuntimeError(f"kdb query failed: {e}") from e

    # Ensure datetime types are sane (PyKX usually does this, but being explicit helps)
    for df, col in [(df_durations, "ts"), (df_latency, "ts"), (df_errors, "ts"), (df_runs, "start_ts"), (df_runs, "end_ts")]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Sort errors newest first
    if not df_errors.empty and "ts" in df_errors.columns:
        df_errors = df_errors.sort_values("ts", ascending=False)

    return {
        "df_durations_all": df_durations,
        "df_latency_all": df_latency,
        "df_errors_all": df_errors,
        "df_runs_all": df_runs,
    }


def q_by_time(df: pd.DataFrame, ts_col: str, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[ts_col, "p50", "p95"])
    out = (
        df.set_index(ts_col)[value_col]
        .resample(RESAMPLE_FREQ)
        .quantile([0.5, 0.95])
        .unstack()
        .rename(columns={0.5: "p50", 0.95: "p95"})
        .reset_index()
    )
    return out


# -----------------------------
# Sidebar: Environment only
# -----------------------------
st.sidebar.header("Environment")
environment = st.sidebar.selectbox("Select environment", ["PROD", "UAT", "DEV"], index=0)
st.sidebar.caption("Coverage: **Today only**")

# Optional: show connection target
cfg = _get_kdb_config()
st.sidebar.caption(f"kdb: `{cfg['host']}:{cfg['port']}`")


# -----------------------------
# Load from kdb
# -----------------------------
try:
    data = load_kdb_all_envs()
except Exception as e:
    st.error("Could not load data from kdb.")
    st.code(str(e))
    st.stop()

df_durations_all = data["df_durations_all"]
df_latency_all = data["df_latency_all"]
df_errors_all = data["df_errors_all"]
df_runs_all = data["df_runs_all"]

# Filter in-memory by env (your requested pattern)
df_durations = df_durations_all[df_durations_all["env"] == environment].copy()
df_latency = df_latency_all[df_latency_all["env"] == environment].copy()
df_errors = df_errors_all[df_errors_all["env"] == environment].copy()
df_runs = df_runs_all[df_runs_all["env"] == environment].copy()


# -----------------------------
# Header
# -----------------------------
st.title("Signals Process Monitor")
st.caption(f"Environment: **{environment}** • Last updated: {NOW:%Y-%m-%d %H:%M:%S} UTC")


# -----------------------------
# KPIs
# -----------------------------
runs_today = len(df_runs)
errors_today = len(df_errors)
success_rate = 100.0 * (df_runs["status"].eq("success").mean() if runs_today else 0.0)

p95_dur = float(df_durations["duration_ms"].quantile(0.95)) if not df_durations.empty else float("nan")
p95_lat = float(df_latency["latency_ms"].quantile(0.95)) if not df_latency.empty else float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Runs today", f"{runs_today:,}")
c2.metric("Success rate", f"{success_rate:.1f}%")
c3.metric("Errors today", f"{errors_today:,}")
c4.metric("P95 function duration (ms)", f"{p95_dur:.0f}" if np.isfinite(p95_dur) else "—")
c5.metric("P95 latency (ms)", f"{p95_lat:.0f}" if np.isfinite(p95_lat) else "—")

st.divider()


# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Function Duration")

    fn_list = sorted(df_durations["function"].dropna().unique().tolist()) if not df_durations.empty else []
    selected_fn = st.selectbox("Function", fn_list, index=0 if fn_list else None)

    if not fn_list:
        st.info("No duration data available for this environment (today).")
    else:
        dff = df_durations[df_durations["function"] == selected_fn]
        chart_df = q_by_time(dff, ts_col="ts", value_col="duration_ms")
        if chart_df.empty:
            st.info("No duration points for the selected function.")
        else:
            st.line_chart(chart_df.set_index("ts")[["p50", "p95"]], height=300)

with right:
    st.subheader("Latency Between Source Tables")

    path_list = sorted(df_latency["path"].dropna().unique().tolist()) if not df_latency.empty else []
    selected_path = st.selectbox("Latency path", path_list, index=0 if path_list else None)

    if not path_list:
        st.info("No latency data available for this environment (today).")
    else:
        dfl = df_latency[df_latency["path"] == selected_path]
        lat_chart = q_by_time(dfl, ts_col="ts", value_col="latency_ms")
        if lat_chart.empty:
            st.info("No latency points for the selected path.")
        else:
            st.line_chart(lat_chart.set_index("ts")[["p50", "p95"]], height=220)

    st.subheader("Errors Logged Today")
    if df_errors.empty:
        st.success("No errors logged today.")
    else:
        # Keep it compact (stacktrace in expander)
        display_cols = [c for c in ["ts", "function", "stage", "message", "host", "run_id"] if c in df_errors.columns]
        st.dataframe(df_errors[display_cols], use_container_width=True, height=220)

        if "stacktrace" in df_errors.columns and df_errors["stacktrace"].notna().any():
            with st.expander("Show most recent stack trace"):
                st.code(str(df_errors.iloc[0].get("stacktrace", "")), language="text")

st.divider()


# -----------------------------
# Runs table
# -----------------------------
st.subheader("Recent Runs")
if df_runs.empty:
    st.info("No runs available for this environment (today).")
else:
    df_runs_view = df_runs.sort_values("start_ts", ascending=False).head(50)
    st.dataframe(df_runs_view, use_container_width=True, height=360)
