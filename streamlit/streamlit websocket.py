import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Must come before any other Streamlit calls
st.set_page_config(layout="wide", page_title="Mock Order Book", page_icon="📊")

# Refresh every second (1000 ms)
st_autorefresh(interval=1000, key="orderbook_refresh")

# ----------------------------
# Mock Order Book Generator
# ----------------------------
def generate_mock_order_book(mid_price: float, spread: float = 0.5, depth: int = 5):
    np.random.seed()  # prevent same randoms on rerun
    bids = []
    asks = []

    for i in range(depth):
        price_bid = round(mid_price - i * spread, 2)
        qty_bid = round(np.random.uniform(1, 10), 2)
        bids.append([price_bid, qty_bid])

        price_ask = round(mid_price + i * spread, 2)
        qty_ask = round(np.random.uniform(1, 10), 2)
        asks.append([price_ask, qty_ask])

    bid_df = pd.DataFrame(bids, columns=["Price", "Qty"])
    ask_df = pd.DataFrame(asks, columns=["Price", "Qty"])
    return bid_df, ask_df

# ----------------------------
# UI + Chart
# ----------------------------

st.title("📡 Mock Real-Time Order Book")

# Symbol dropdown
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AAPL/USD"]
selected_symbol = st.selectbox("Select Symbol", symbols)

# Generate random mid price for selected symbol
base_prices = {"BTC/USDT": 30000, "ETH/USDT": 2000, "SOL/USDT": 100, "AAPL/USD": 180}
mid_price = base_prices[selected_symbol] + np.random.uniform(-5, 5)

# Generate mock order book
bids_df, asks_df = generate_mock_order_book(mid_price)

# Layout: Tables
col1, col2 = st.columns(2)

col1.subheader("💰 Top Bids")
col1.table(bids_df)

col2.subheader("🧾 Top Asks")
col2.table(asks_df)

# Layout: Depth Chart
st.subheader("📊 Depth Chart (Price vs Cumulative Qty)")

bids_df_sorted = bids_df.sort_values(by="Price", ascending=True)
asks_df_sorted = asks_df.sort_values(by="Price", ascending=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=bids_df_sorted["Price"],
    y=bids_df_sorted["Qty"].cumsum(),
    mode="lines+markers",
    name="Bids",
    fill="tozeroy"
))

fig.add_trace(go.Scatter(
    x=asks_df_sorted["Price"],
    y=asks_df_sorted["Qty"].cumsum(),
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