import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import plotly.graph_objects as go
from threading import Thread

# Global order book cache
order_book_data = {"bids": pd.DataFrame(), "asks": pd.DataFrame()}
available_symbols = ["btcusdt", "ethusdt", "bnbusdt", "solusdt"]  # You can add more here

# Binance WebSocket async function
async def binance_depth_ws(symbol):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth5@100ms"
    async with websockets.connect(url) as websocket:
        while True:
            data = await websocket.recv()
            parsed = json.loads(data)

            bids = pd.DataFrame(parsed["bids"], columns=["Price", "Qty"]).astype(float)
            asks = pd.DataFrame(parsed["asks"], columns=["Price", "Qty"]).astype(float)

            # Sort and keep top 5 levels
            order_book_data["bids"] = bids.sort_values("Price", ascending=False).head(5)
            order_book_data["asks"] = asks.sort_values("Price", ascending=True).head(5)

# Wrapper for asyncio thread
def start_ws(symbol):
    asyncio.run(binance_depth_ws(symbol))

# Launch WebSocket in background thread
def launch_ws_thread(symbol):
    thread = Thread(target=start_ws, args=(symbol,), daemon=True)
    thread.start()

# Streamlit page config
st.set_page_config(layout="wide")
st.title("📡 Real-Time Order Book Viewer")

# Symbol selection
symbol = st.selectbox("Choose trading pair", available_symbols)

# Restart WebSocket if symbol changes
if "current_symbol" not in st.session_state or st.session_state.current_symbol != symbol:
    st.session_state.current_symbol = symbol
    order_book_data["bids"], order_book_data["asks"] = pd.DataFrame(), pd.DataFrame()
    launch_ws_thread(symbol)

# Live update display
placeholder = st.empty()

while True:
    with placeholder.container():
        col1, col2 = st.columns(2)

        # Order Book Tables
        col1.subheader("💰 Bids")
        col1.table(order_book_data["bids"])

        col2.subheader("🧾 Asks")
        col2.table(order_book_data["asks"])

        # Depth Chart
        st.subheader("📊 Depth Chart (Price vs Qty)")

        if not order_book_data["bids"].empty and not order_book_data["asks"].empty:
            bid_prices = order_book_data["bids"]["Price"]
            bid_qtys = order_book_data["bids"]["Qty"].cumsum()  # Cumulative size

            ask_prices = order_book_data["asks"]["Price"]
            ask_qtys = order_book_data["asks"]["Qty"].cumsum()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=bid_prices, y=bid_qtys,
                mode='lines+markers',
                name='Bids',
                fill='tozeroy'
            ))

            fig.add_trace(go.Scatter(
                x=ask_prices, y=ask_qtys,
                mode='lines+markers',
                name='Asks',
                fill='tozeroy'
            ))

            fig.update_layout(
                xaxis_title='Price',
                yaxis_title='Cumulative Quantity',
                template='plotly_white',
                height=400,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

    st.sleep(1)