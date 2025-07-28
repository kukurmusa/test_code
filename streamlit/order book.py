import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from qpython import qconnection

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="L2 Order Book Playback", page_icon="📈")
st.title("📉 L2 Order Book Playback")

# --- GET QUERY PARAMS FROM URL ---
query_params = st.query_params
input_date = datetime.today().date()
input_sym = "BTCUSD"
input_clordid = ""

if "date" in query_params:
    try:
        input_date = datetime.strptime(query_params["date"], "%Y-%m-%d").date()
    except:
        pass

if "sym" in query_params:
    input_sym = query_params["sym"]

# --- SIDEBAR INPUT ---
with st.sidebar:
    st.header("🔎 Symbol & Filters")
    input_date = st.date_input("Select Date", input_date)
    input_sym = st.text_input("Enter Symbol", input_sym)
    input_clordid = st.text_input("Enter ClOrdID (optional)", input_clordid)
    run_button = st.button("🔄 Run Analysis", type="primary")

# --- CACHED FUNCTION TO SIMULATE KDB RESPONSE ---
@st.cache_data(show_spinner=True)
def simulate_kdb_response(date, sym):
    q = qconnection.QConnection(host="ldnqlpatcqa901", port=12347, username="eueqt", password="eueqt", pandas=True)
    q.open()
    
    query = f"""
    .file.loadScript[system "getenv HOME", "/svn/algo/trunk/q/util/LV2_util.q"];
    X: queryX[`{date.strftime("%Y.%m.%d")}; "{sym}"];
    """
    L2Book = q(query)
    q.close()

    orderBook = L2Book["orderBook"]
    rawOrderBooks = L2Book["rawOrderBooks"]
    
    return {
        "orderBook": orderBook,
        "rawOrderBooks": rawOrderBooks
    }

# --- INITIALISE STATE ---
if "order_book" not in st.session_state:
    st.session_state.order_book = {}
if "order_book_history" not in st.session_state:
    st.session_state.order_book_history = None
if "current_timestamp" not in st.session_state:
    st.session_state.current_timestamp = None
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# --- ON BUTTON CLICK: LOAD NEW DATA ---
if run_button:
    st.session_state.order_book = simulate_kdb_response(input_date, input_clordid)
    st.session_state.order_book_history = st.session_state.order_book["rawOrderBooks"]
    st.session_state.current_timestamp = st.session_state.order_book_history["time"].iloc[0]

# --- EXIT IF NO DATA ---
if st.session_state.order_book_history is None:
    st.info("Please enter inputs and run analysis.")
    st.stop()

# --- TIMESTAMP SLIDER SETUP ---
timestamps = st.session_state.order_book_history["time"].tolist()
timestamp_strs = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]

# Set slider default if needed
if st.session_state.current_timestamp not in timestamps:
    st.session_state.current_timestamp = timestamps[0]

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
bookData = st.session_state.order_book_history
current_time = st.session_state.current_timestamp

book_snapshot = bookData[bookData["time"] == current_time]

bids_df = book_snapshot[(book_snapshot["side"] == "B") & (book_snapshot["price"] > 0)].sort_values("price", ascending=False).head(20)
asks_df = book_snapshot[(book_snapshot["side"] == "S") & (book_snapshot["price"] > 0)].sort_values("price", ascending=True).head(20)

# --- DISPLAY TABLES ---
col1, col2 = st.columns(2)
col1.subheader(f"💰 Top Bid Book @ {current_time.strftime('%H:%M:%S')}")
col1.table(bids_df[["price", "size"]])

col2.subheader(f"🧾 Top Ask Book @ {current_time.strftime('%H:%M:%S')}")
col2.table(asks_df[["price", "size"]])

# --- DEPTH CHART ---
st.subheader("📊 Depth Chart")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=bids_df.sort_values("price")["price"],
    y=bids_df.sort_values("price")["size"].cumsum(),
    mode="lines+markers",
    name="Bids",
    fill="tozeroy"
))
fig.add_trace(go.Scatter(
    x=asks_df.sort_values("price")["price"],
    y=asks_df.sort_values("price")["size"].cumsum(),
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
