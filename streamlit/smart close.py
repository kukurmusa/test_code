import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Simulated Data ----
np.random.seed(42)
time_index = pd.date_range("2024-01-01 09:30", periods=60, freq="1min")
components = ["ChildA", "ChildB", "ChildC"]

def generate_timeseries(name):
    return pd.DataFrame({
        "timestamp": np.tile(time_index, len(components)),
        "component": np.repeat(components, len(time_index)),
        name: np.random.rand(len(time_index) * len(components)) * 100
    })

def simulate_kdb_response(date, clordid):
    return {
        "order_details": pd.DataFrame({
            "ClOrdID": [clordid],
            "Date": [date],
            "Symbol": ["AAPL"],
            "OrderQty": [10000],
            "ExecQty": [8750],
            "AvgPx": [150.25]
        }),
        "benchmark_data": pd.DataFrame({
            "Benchmark": ["Arrival", "VWAP", "TWAP"],
            "CostBps": [5.2, 3.8, 4.1]
        }),
        "execution_summary": pd.DataFrame({
            "Component": components,
            "StartTime": [time_index[0]] * 3,
            "EndTime": [time_index[-1]] * 3,
            "FillQty": [3000, 3500, 2250]
        }),
        "quantity_data": generate_timeseries("quantity"),
        "price_data": generate_timeseries("price"),
        "trigger_data": generate_timeseries("trigger_level"),
        "l2_data": generate_timeseries("l2_bid_size").rename(columns={"l2_bid_size": "value"})
    }

# ---- Streamlit UI ----
st.set_page_config(page_title="Algo Order Tracker", layout="wide")
st.title("🧠 Algo Order Performance Tracker")

# Inputs
with st.sidebar:
    st.header("Order Filters")
    input_date = st.date_input("Select Date", datetime.today())
    input_clordid = st.text_input("Enter ClOrdID", "ABC123XYZ")

@st.cache_data
def fetch_kdb_data(date, clordid):
    return simulate_kdb_response(date, clordid)

data = fetch_kdb_data(input_date, input_clordid)

# ---- Tables ----
st.subheader("Order Details")
st.dataframe(data["order_details"], use_container_width=True)

st.subheader("Benchmark Performance")
st.dataframe(data["benchmark_data"], use_container_width=True)

st.subheader("Execution Summary")
st.dataframe(data["execution_summary"], use_container_width=True)

# ---- Synced Charts ----
st.subheader("Algo Component Time Series")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=["Quantity", "Price", "Trigger Level"])

for comp in components:
    q = data["quantity_data"].query("component == @comp")
    p = data["price_data"].query("component == @comp")
    t = data["trigger_data"].query("component == @comp")
    fig.add_trace(go.Scatter(x=q.timestamp, y=q.quantity, name=f"Qty - {comp}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.price, name=f"Price - {comp}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t.timestamp, y=t.trigger_level, name=f"Trigger - {comp}"), row=3, col=1)

fig.update_layout(height=900, showlegend=True, title="Synced Time Series")
st.plotly_chart(fig, use_container_width=True)

# ---- L2 Data ----
st.subheader("L2 Market Data")

l2_fig = go.Figure()
for comp in components:
    l2 = data["l2_data"].query("component == @comp")
    l2_fig.add_trace(go.Scatter(x=l2.timestamp, y=l2.value, name=f"L2 Bid - {comp}"))

l2_fig.update_layout(title="L2 Bid Size Over Time")
st.plotly_chart(l2_fig, use_container_width=True)
