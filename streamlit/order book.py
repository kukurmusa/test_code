import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Mock Order Book Playback", page_icon="🎬")

st.title("🎬 Order Book Playback Viewer")

# --- SYMBOL DROPDOWN ---
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AAPL/USD"]
selected_symbol = st.selectbox("Select Symbol", symbols)

# --- STATE INITIALISATION ---
if "symbol" not in st.session_state:
    st.session_state.symbol = None
if "order_book_history" not in st.session_state:
    st.session_state.order_book_history = []
if "time_index" not in st.session_state:
    st.session_state.time_index = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# --- GENERATE MOCK DATA IF SYMBOL CHANGED ---
def generate_order_book_history(base_price, steps=20, spread=0.5, depth=5):
    history = []
    for _ in range(steps):
        mid_price = base_price + np.random.normal(0, 2)
        bids, asks = [], []
        for i in range(depth):
            bids.append([round(mid_price - i * spread, 2), round(np.random.uniform(1, 10), 2)])
            asks.append([round(mid_price + i * spread, 2), round(np.random.uniform(1, 10), 2)])
        bid_df = pd.DataFrame(bids, columns=["Price", "Qty"])
        ask_df = pd.DataFrame(asks, columns=["Price", "Qty"])
        history.append((bid_df, ask_df))
    return history

base_prices = {"BTC/USDT": 30000, "ETH/USDT": 2000, "SOL/USDT": 100, "AAPL/USD": 180}

if st.session_state.symbol != selected_symbol:
    st.session_state.symbol = selected_symbol
    st.session_state.order_book_history = generate_order_book_history(base_prices[selected_symbol])
    st.session_state.time_index = 0
    st.session_state.is_playing = False

max_steps = len(st.session_state.order_book_history) - 1

# --- PLAY/PAUSE BUTTONS ---
col_play, col_slider = st.columns([1, 5])
with col_play:
    if st.session_state.is_playing:
        if st.button("⏸️ Pause"):
            st.session_state.is_playing = False
    else:
        if st.button("▶️ Play"):
            st.session_state.is_playing = True

# --- TIME SLIDER ---
with col_slider:
    st.session_state.time_index = st.slider(
        "Select Time Step",
        0,
        max_steps,
        value=st.session_state.time_index,
        step=1,
        key="slider_time_index"
    )

# --- ANIMATION LOOP ---
if st.session_state.is_playing:
    st.session_state.time_index += 1
    if st.session_state.time_index > max_steps:
        st.session_state.time_index = 0  # Loop around
    time.sleep(1)
    st.experimental_rerun()

# --- DISPLAY SNAPSHOT ---
bids_df, asks_df = st.session_state.order_book_history[st.session_state.time_index]

col1, col2 = st.columns(2)
col1.subheader(f"💰 Top Bids (Time {st.session_state.time_index})")
col1.table(bids_df)

col2.subheader(f"🧾 Top Asks (Time {st.session_state.time_index})")
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