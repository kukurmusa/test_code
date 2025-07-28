import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Mock Order Book Playback", page_icon="🕒")

st.title("🕒 Order Book Playback (Timestamp-based)")

# --- SYMBOL DROPDOWN ---
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AAPL/USD"]
selected_symbol = st.selectbox("Select Symbol", symbols)

# --- STATE INITIALISATION ---
if "symbol" not in st.session_state:
    st.session_state.symbol = None
if "order_book_history" not in st.session_state:
    st.session_state.order_book_history = {}
if "current_timestamp" not in st.session_state:
    st.session_state.current_timestamp = None
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# --- GENERATE MOCK TIMESTAMPED ORDER BOOK HISTORY ---
def generate_order_book_history(base_price, steps=20, interval_seconds=60, spread=0.5, depth=5):
    history = {}
    start_time = datetime.now().replace(second=0, microsecond=0)
    for i in range(steps):
        ts = start_time + timedelta(seconds=i * interval_seconds)
        mid_price = base_price + np.random.normal(0, 2)
        bids, asks = [], []
        for j in range(depth):
            bids.append([round(mid_price - j * spread, 2), round(np.random.uniform(1, 10), 2)])
            asks.append([round(mid_price + j * spread, 2), round(np.random.uniform(1, 10), 2)])
        bid_df = pd.DataFrame(bids, columns=["Price", "Qty"])
        ask_df = pd.DataFrame(asks, columns=["Price", "Qty"])
        history[ts] = (bid_df, ask_df)
    return history

# If symbol changed, regenerate
base_prices = {"BTC/USDT": 30000, "ETH/USDT": 2000, "SOL/USDT": 100, "AAPL/USD": 180}

if st.session_state.symbol != selected_symbol:
    st.session_state.symbol = selected_symbol
    st.session_state.order_book_history = generate_order_book_history(base_prices[selected_symbol])
    st.session_state.current_timestamp = list(st.session_state.order_book_history.keys())[0]
    st.session_state.is_playing = False

timestamps = sorted(list(st.session_state.order_book_history.keys()))
timestamp_strs = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
current_index = timestamps.index(st.session_state.current_timestamp)

# --- PLAY/PAUSE BUTTONS ---
col_play, col_slider = st.columns([1, 5])
with col_play:
    if st.session_state.is_playing:
        if st.button("⏸️ Pause"):
            st.session_state.is_playing = False
    else:
        if st.button("▶️ Play"):
            st.session_state.is_playing = True

# --- TIMESTAMP SLIDER ---
with col_slider:
    selected_str = st.select_slider(
        "Select Timestamp",
        options=timestamp_strs,
        value=st.session_state.current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        key="timestamp_slider"
    )
    st.session_state.current_timestamp = datetime.strptime(selected_str, "%Y-%m-%d %H:%M:%S")

# --- ANIMATION LOOP ---
if st.session_state.is_playing:
    next_index = current_index + 1
    if next_index >= len(timestamps):
        next_index = 0  # loop
    st.session_state.current_timestamp = timestamps[next_index]
    time.sleep(1)
    st.experimental_rerun()

# --- DISPLAY ORDER BOOK SNAPSHOT ---
bids_df, asks_df = st.session_state.order_book_history[st.session_state.current_timestamp]

col1, col2 = st.columns(2)
col1.subheader(f"💰 Top Bids ({st.session_state.current_timestamp.strftime('%H:%M:%S')})")
col1.table(bids_df)

col2.subheader(f"🧾 Top Asks ({st.session_state.current_timestamp.strftime('%H:%M:%S')})")
col2.table(asks_df)

# --- DEPTH CHART ---
st.subheader("📊 Depth Chart (Price vs Cumulative Qty)")

bids_sorted = bids_df.sort_values("Price", ascending=True)
asks_sorted = asks_df.sort_values("Price", ascending=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=bids_sorted["Price"],
    y=bids_sorted["Qty"].cumsum(),
    mode="lines+markers",
    name="Bids",
    fill="tozeroy"
))
fig.add_trace(go.Scatter(
    x=asks_sorted["Price"],
    y=asks_sorted["Qty"].cumsum(),
    mode="lines+markers",
    name="Asks",
    fill="tozeroy"
))
fig.update_layout(
    xaxis_title="Price",
    yaxis_title="Cumulative Quantity",
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)