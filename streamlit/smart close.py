import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Simulated kdb+ Response ----
np.random.seed(42)
time_index = pd.date_range("2024-01-01 09:30", periods=60, freq="1min")

def generate_timeseries(name):
    components = ["ChildA", "ChildB", "ChildC"]
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
            "Component": ["ChildA", "ChildB", "ChildC"],
            "StartTime": [time_index[0]] * 3,
            "EndTime": [time_index[-1]] * 3,
            "FillQty": [3000, 3500, 2250]
        }),
        "quantity_data": generate_timeseries("quantity"),
        "price_data": generate_timeseries("price"),
        "trigger_data": generate_timeseries("trigger_level"),
        "l2_data": generate_timeseries("l2_bid_size").rename(columns={"l2_bid_size": "value"})
    }

# ---- Caching ----
@st.cache_data
def fetch_kdb_data(date, clordid):
    return simulate_kdb_response(date, clordid)

# ---- Page Config ----
st.set_page_config(page_title="Algo Order Tracker", layout="wide")
st.title("🧠 Algo Order Performance Tracker")

# ---- Read Query Params ----
query_params = st.query_params
input_date = query_params.get("date", [str(datetime.today().date())])[0]
input_clordid = query_params.get("clordid", [""])[0]
input_date = datetime.strptime(input_date, "%Y-%m-%d").date()

# ---- Sidebar Inputs ----
with st.sidebar:
    st.header("Order Filters")
    input_date = st.date_input("Select Date", input_date)
    input_clordid = st.text_input("Enter ClOrdID", input_clordid)
    run_button = st.button("Run Analysis", type="primary")

# ---- Main Analysis ----
if run_button:
    # Set query params in URL
    st.query_params["date"] = str(input_date)
    st.query_params["clordid"] = input_clordid

    data = fetch_kdb_data(input_date, input_clordid)

    st.subheader("Order Details")
    st.dataframe(data["order_details"], use_container_width=True)

    st.subheader("Benchmark Performance")
    st.dataframe(data["benchmark_data"], use_container_width=True)

    st.subheader("Execution Summary")
    st.dataframe(data["execution_summary"], use_container_width=True)

    # ---- Synced Time Series Charts ----
    st.subheader("Algo Component Time Series")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=["Quantity", "Price", "Trigger Level"])

    for comp in data["quantity_data"]["component"].unique():
        df = data["quantity_data"].query("component == @comp")
        fig.add_trace(go.Scatter(x=df.timestamp, y=df.quantity, name=f"Qty - {comp}"), row=1, col=1)

    for comp in data["price_data"]["component"].unique():
        df = data["price_data"].query("component == @comp")
        fig.add_trace(go.Scatter(x=df.timestamp, y=df.price, name=f"Price - {comp}"), row=2, col=1)

    for comp in data["trigger_data"]["component"].unique():
        df = data["trigger_data"].query("component == @comp")
        fig.add_trace(go.Scatter(x=df.timestamp, y=df.trigger_level, name=f"Trigger - {comp}"), row=3, col=1)

    fig.update_layout(height=900, showlegend=True, title="Synced Algo Component Charts")
    st.plotly_chart(fig, use_container_width=True)

    # ---- L2 Market Data ----
    st.subheader("L2 Market Data")

    l2_fig = go.Figure()
    for comp in data["l2_data"]["component"].unique():
        df = data["l2_data"].query("component == @comp")
        l2_fig.add_trace(go.Scatter(x=df.timestamp, y=df.value, name=f"L2 - {comp}"))

    l2_fig.update_layout(title="L2 Bid Size Over Time")
    st.plotly_chart(l2_fig, use_container_width=True)
else:
    st.info("Please enter inputs and click **Run Analysis** to load order data.")
