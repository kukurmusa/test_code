import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Order Book Playback", page_icon="📈")

st.title("📈 BTC/USDT Order Book Playback")

# --- STATE INITIALISATION ---
if "order_book_history" not in st.session_state:
    st.session_state.order_book_history = {}
if "current_timestamp" not in st.session_state:
    st.session_state.current_timestamp = None
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# --- GENERATE MOCK TIMESTAMPED ORDER BOOK ---
def generate_order_book_history(base_price=30000, steps=20, interval_seconds=60, spread=1.0, depth=5):
    history = {}
    start_time = datetime.now().replace(second=0, microsecond=0)
    for i in range(steps):
        ts = start_time + timedelta(seconds=i * interval_seconds)
        mid_price = base_price + np.random.normal(0, 10)
        bids, asks = [], []
        for j in range(depth):
            bids.append([round(mid_price - j * spread, 2), round(np.random.uniform(1, 10), 2)])
            asks.append([round(mid_price + j * spread, 2), round(np.random.uniform(1, 10), 2)])
        bid_df = pd.DataFrame(bids, columns=["Price", "Qty"])
        ask_df = pd.DataFrame(asks, columns=["Price", "Qty"])
        history[ts] = (bid_df, ask_df)
    return history

# Generate once
if not st.session_state.order_book_history:
    st.session_state.order_book_history = generate_order_book_history()
    st.session_state.current_timestamp = list(st.session_state.order_book_history.keys())[0]

timestamps = sorted(st.session_state.order_book_history.keys())
timestamp_strs = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
current_index = timestamps.index(st.session_state.current_timestamp)

# --- PLAY/PAUSE CONTROLS ---
col1, col2 = st.columns([1, 5])

with col1:
    if st.session_state.is_playing:
        if st.button("⏸️ Pause"):
            st.session_state.is_playing = False
    else:
        if st.button("▶️ Play"):
            st.session_state.is_playing = True

with col2:
    selected_str = st.select_slider(
        "Select Timestamp",
        options=timestamp_strs,
        value=st.session_state.current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    )
    st.session_state.current_timestamp = datetime.strptime(selected_str, "%Y-%m-%d %H:%M:%S")

# --- PLAYBACK ANIMATION ---
if st.session_state.is_playing:
    next_index = (current_index + 1) % len(timestamps)
    st.session_state.current_timestamp = timestamps[next_index]
    time.sleep(1)
    st.experimental_rerun()

# --- DISPLAY ORDER BOOK SNAPSHOT ---
bids_df, asks_df = st.session_state.order_book_history[st.session_state.current_timestamp]

col1, col2 = st.columns(2)
col1.subheader(f"💰 Top Bids @ {st.session_state.current_timestamp.strftime('%H:%M:%S')}")
col1.table(bids_df)

col2.subheader(f"🧾 Top Asks @ {st.session_state.current_timestamp.strftime('%H:%M:%S')}")
col2.table(asks_df)

# --- DEPTH CHART ---
st.subheader("📊 Depth Chart")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=bids_df.sort_values("Price")["Price"],
    y=bids_df.sort_values("Price")["Qty"].cumsum(),
    mode="lines+markers",
    name="Bids",
    fill="tozeroy"
))
fig.add_trace(go.Scatter(
    x=asks_df.sort_values("Price")["Price"],
    y=asks_df.sort_values("Price")["Qty"].cumsum(),
    mode="lines+markers",
    name="Asks",
    fill="tozeroy"
))
fig.update_layout(
    xaxis_title="Price",
    yaxis_title="Cumulative Quantity",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig, use_container_width=True)