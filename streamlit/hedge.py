import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import random
import time

# Dummy function to generate 1-year daily returns
def generate_dummy_returns(seed=0):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=252)
    returns = np.random.normal(loc=0, scale=0.01, size=len(dates)).cumsum()
    return pd.Series(returns, index=dates)

def query_kdb_for_basket(query_kdb_for_basket):
    # Dummy KDB data for demonstration
    data = {
        "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "Position ($)": [150_000, -80_000, 120_000, -100_000, 50_000],
        "Direction": ["Long", "Short", "Long", "Short", "Long"]
    }
    return pd.DataFrame(data)

def simulate_hedge_performance(basket_returns, hedge_returns, basket_beta, hedge_beta):
    # Basic hedge ratio
    hedge_ratio = basket_beta / hedge_beta if hedge_beta != 0 else 0

    # Hedged returns
    hedged_returns = basket_returns - hedge_ratio * hedge_returns

    # Tracking error
    tracking_error = np.std(basket_returns - hedged_returns)

    # Cumulative returns
    basket_cum = basket_returns.cumsum()
    hedge_cum = hedge_returns.cumsum()
    hedged_cum = hedged_returns.cumsum()

    return hedged_returns, tracking_error, basket_cum, hedge_cum, hedged_cum

def get_detailed_hedge_recommendations(basket, basket_notional):
    hedge_data = [
        {"Instrument": "S&P 500 Futures (ES)", "Type": "Index Future", "Price": 5200, "ContractSize": 50, "Beta": 1.05,
         "Correlation": 0.85},
        {"Instrument": "FTSE 100 Futures (Z)", "Type": "Index Future", "Price": 8400, "ContractSize": 10, "Beta": 0.95,
         "Correlation": 0.75},
        {"Instrument": "SPY ETF", "Type": "ETF", "Price": 500, "ContractSize": 1, "Beta": 1.00, "Correlation": 0.82},
        {"Instrument": "XLF ETF (Financials)", "Type": "Sector ETF", "Price": 40, "ContractSize": 1, "Beta": 1.10,
         "Correlation": 0.65},
        {"Instrument": "XLK ETF (Technology)", "Type": "Sector ETF", "Price": 200, "ContractSize": 1, "Beta": 1.20,
         "Correlation": 0.78},
        {"Instrument": "SH (Inverse S&P 500 ETF)", "Type": "Inverse ETF", "Price": 30, "ContractSize": 1, "Beta": -1.00,
         "Correlation": -0.85},
        {"Instrument": "GLD ETF (Gold)", "Type": "Commodity ETF", "Price": 190, "ContractSize": 1, "Beta": 0.10,
         "Correlation": 0.20},
        {"Instrument": "TLT ETF (Long-Term US Treasuries)", "Type": "Bond ETF", "Price": 100, "ContractSize": 1,
         "Beta": -0.05, "Correlation": -0.25},
        {"Instrument": "VIX Futures", "Type": "Volatility Future", "Price": 20, "ContractSize": 1000, "Beta": -0.50,
         "Correlation": -0.55},
        {"Instrument": "Single-Stock Hedge Basket (AAPL, MSFT, GOOGL, AMZN, META)", "Type": "Basket Hedge", "Price": 0,
         "ContractSize": 1, "Beta": 1.10, "Correlation": 0.72}
    ]

    recommendations = []
    for hedge in hedge_data:
        hedge_notional = basket_notional / hedge["Beta"] if hedge["Beta"] != 0 else 0
        num_contracts = hedge_notional / (hedge["Price"] * hedge["ContractSize"]) if hedge["Price"] > 0 else 0

        adv_percent = round(random.uniform(5, 15), 2)
        pov_percent = round(random.uniform(10, 30), 2)

        recommendations.append({
            "Hedge Instrument": hedge["Instrument"],
            "Type": hedge["Type"],
            "Price": hedge["Price"],
            "Contract Size": hedge["ContractSize"],
            "Beta": hedge["Beta"],
            "Correlation": hedge["Correlation"],
            "Num Contracts/Shares": round(num_contracts, 2),
            "Hedge $ Value": round(hedge["Price"] * hedge["ContractSize"] * num_contracts, 2) if hedge["Price"] > 0 else basket_notional,
            "ADV%": adv_percent,
            "Expected POV%": pov_percent,
            "Description": f"Hedge with {hedge['Instrument']} ({hedge['Type']})."
        })

    return pd.DataFrame(recommendations)

# App layout
st.set_page_config(page_title="Hedge Recommendations", layout="wide")
st.title("🛡️ Interactive Hedge Recommendations")

# File upload and validation
# uploaded_file = st.file_uploader("Upload your basket CSV file here (Ticker, Position ($), Direction):", type=["csv"])

data_source = st.radio(
    "Select data source for basket positions:",
    ("Upload CSV File", "Query from KDB"),
    index=0
)
basket_df = None

if data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload your basket CSV file:", type=["csv"])
    if uploaded_file is not None:
        basket_df = pd.read_csv(uploaded_file)
        st.session_state["basket_df"] = basket_df
        # st.dataframe(basket_df)
elif data_source == "Query from KDB":
    query_string = st.text_area("Enter your KDB query:", "select from basket where ...")  # Let user input query
    if st.button("Run KDB Query"):
        basket_df = query_kdb_for_basket(query_string)
        st.success("✅ Basket data loaded from KDB!")
        st.session_state["basket_df"] = basket_df
        # st.dataframe(basket_df)
    # st.dataframe(basket_df)

if "basket_df" in st.session_state:
    basket_df = st.session_state["basket_df"]
else:
    basket_df = None

if basket_df is not None:
    # with st.spinner("Processing uploaded file..."):
    #     time.sleep(2)

    required_columns = {"Ticker", "Position ($)", "Direction"}
    if not required_columns.issubset(basket_df.columns):
        st.error("Loaded basket is missing required columns: Ticker, Position ($), Direction.")
    else:
        st.success("✅ Basket file loaded and validated!")

        with st.expander("📂 Basket Positions", expanded=True):
            gb = GridOptionsBuilder.from_dataframe(basket_df)
            paginationSize = min(len(basket_df),10)
            gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=paginationSize)
            gb.configure_default_column(editable=False, groupable=True)

            gb.configure_column(
                "Position ($)",
                type=["numericColumn"],
                valueFormatter="x.toLocaleString('en-US')"
            )

            grid_options = gb.build()

            # Calculate dynamic height for 10 rows per page
            row_height = 35  # Adjust based on font/row spacing
            calculated_height = row_height * (paginationSize + 1)  # 10 rows + header

            AgGrid(
                basket_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.NO_UPDATE,
                height=calculated_height,  # Dynamically calculated
                width='100%',
                fit_columns_on_grid_load=True
            )

        with st.expander("📊 Pre-Trade Summary", expanded=True):
            long_positions = basket_df[basket_df["Direction"].str.lower() == "long"]["Position ($)"].sum()
            short_positions = basket_df[basket_df["Direction"].str.lower() == "short"]["Position ($)"].sum()
            gross_notional = long_positions + abs(short_positions)
            net_notional = long_positions - abs(short_positions)

            tab1, tab2, tab3 = st.tabs(["Overall", "Country", "Sector"])

            overall_summary = pd.DataFrame({
                "Metric": ["Total Gross Notional ($)", "Net Notional ($)", "Number of Long Positions", "Number of Short Positions"],
                "Value": [f"${gross_notional:,.2f}", f"${net_notional:,.2f}", f"{basket_df[basket_df['Direction'].str.lower() == 'long'].shape[0]}", f"{basket_df[basket_df['Direction'].str.lower() == 'short'].shape[0]}"]
            })

            with tab1:
                st.dataframe(overall_summary, use_container_width=True)
            with tab2:
                st.dataframe(pd.DataFrame({"Country": ["US", "UK", "EU"], "Exposure ($)": [f"${gross_notional*0.5:,.2f}", f"${gross_notional*0.3:,.2f}", f"${gross_notional*0.2:,.2f}"]}), use_container_width=True)
            with tab3:
                st.dataframe(pd.DataFrame({"Sector": ["Tech", "Financials", "Energy"], "Exposure ($)": [f"${gross_notional*0.4:,.2f}", f"${gross_notional*0.4:,.2f}", f"${gross_notional*0.2:,.2f}"]}), use_container_width=True)

        with st.expander("🗂️ Hedge Recommendations Grid", expanded=True):
            hedge_df = get_detailed_hedge_recommendations(basket_df["Ticker"].tolist(), gross_notional)

            hedge_types = hedge_df["Type"].unique()
            hedge_tabs = st.tabs(list(hedge_types))

            for i, hedge_type in enumerate(hedge_types):
                with hedge_tabs[i]:
                    st.markdown(f"### Hedge Type: {hedge_type}")
                    filtered_hedge_df = hedge_df[hedge_df["Type"] == hedge_type]

                    # Format & style
                    format_dict = {"Price": "{:,.2f}", "Contract Size": "{:,.0f}", "Beta": "{:,.2f}",
                                   "Correlation": "{:,.2f}",
                                   "Num Contracts/Shares": "{:,.2f}", "Hedge $ Value": "${:,.2f}",
                                   "ADV%": "{:,.2f}%", "Expected POV%": "{:,.2f}%"}
                    styled_df = filtered_hedge_df.style.format(format_dict)
                    st.dataframe(styled_df, use_container_width=True)

                    # Dropdown for selected hedge
                    hedge_instruments = filtered_hedge_df["Hedge Instrument"].tolist()
                    selected_hedge_instrument = st.selectbox(
                        f"**Select a {hedge_type} Hedge Instrument:**",
                        hedge_instruments,
                        key=f"hedge_select_{hedge_type}"
                    )

                    if selected_hedge_instrument:
                        selected_hedge = hedge_df[hedge_df["Hedge Instrument"] == selected_hedge_instrument].iloc[0]
                        tab1, tab2 = st.tabs(["Text Details", "Charts"])

                        with tab1:
                            if "Single-Stock Hedge Basket" in selected_hedge_instrument:
                                hedge_basket_df = pd.DataFrame({
                                    "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                                    "Position ($)": [100_000, 80_000, 90_000, 70_000, 60_000],
                                    "Direction": ["Long"] * 5
                                })
                                hedge_summary = pd.DataFrame(
                                    {"Metric": ["Total Hedge Notional ($)", "Number of Positions"],
                                     "Value": [f"${hedge_basket_df['Position ($)'].sum():,.2f}", "5"]})
                                st.write("### 📦 Hedge Basket Constituents")
                                st.dataframe(hedge_basket_df, use_container_width=True)
                                st.write("### 🧮 Hedge Basket Pre-Trade Summary")
                                st.dataframe(hedge_summary, use_container_width=True)

                            st.write(f"**Beta:** {selected_hedge['Beta']:,.2f}")
                            st.write(f"**Correlation:** {selected_hedge['Correlation']:,.2f}")
                            st.write(f"**ADV%:** {selected_hedge['ADV%']:,.2f}%")
                            st.write(f"**Expected POV%:** {selected_hedge['Expected POV%']:,.2f}%")
                            st.write(
                                f"**Total Hedge $ Value:** ${selected_hedge['Hedge $ Value']:,.2f} (for 100% basket hedge)")
                        with tab2:
                            basket_returns = generate_dummy_returns(seed=42)
                            hedge_returns = generate_dummy_returns(
                                seed=999 if "Single-Stock Hedge Basket" in selected_hedge_instrument else hedge_instruments.index(
                                    selected_hedge_instrument))
                            fig_returns = go.Figure()
                            fig_returns.add_trace(
                                go.Scatter(x=basket_returns.index, y=basket_returns.values, mode='lines',
                                           name='Basket Returns', line=dict(width=2)))
                            fig_returns.add_trace(
                                go.Scatter(x=hedge_returns.index, y=hedge_returns.values, mode='lines',
                                           name=f'{selected_hedge_instrument} Returns',
                                           line=dict(width=2, dash='dash')))
                            fig_returns.update_layout(title="Cumulative Dummy Returns", xaxis_title="Date",
                                                      yaxis_title="Cumulative Returns")
                            st.plotly_chart(fig_returns, use_container_width=True)

                            # Simulate hedged basket performance
                            hedged_returns, tracking_error, basket_cum, hedge_cum, hedged_cum = simulate_hedge_performance(
                                basket_returns,
                                hedge_returns,
                                basket_beta=1.0,  # or real basket beta
                                hedge_beta=selected_hedge['Beta']
                            )

                            # Plot cumulative returns
                            fig_cum = go.Figure()
                            fig_cum.add_trace(
                                go.Scatter(x=basket_returns.index, y=basket_cum, name="Basket", line=dict(width=2)))
                            fig_cum.add_trace(go.Scatter(x=hedge_returns.index, y=hedge_cum, name="Hedge Instrument",
                                                         line=dict(width=2, dash='dash')))
                            fig_cum.add_trace(go.Scatter(x=hedge_returns.index, y=hedged_cum, name="Hedged Basket",
                                                         line=dict(width=2, dash='dot')))
                            fig_cum.update_layout(title="Cumulative Returns (Simulated Hedge Impact)",
                                                  xaxis_title="Date",
                                                  yaxis_title="Cumulative Returns")
                            st.plotly_chart(fig_cum, use_container_width=True)

                            # Display tracking error
                            st.write(f"**Simulated Tracking Error:** {tracking_error:.4f}")

        # with st.expander("🔍 Hedge Details and Chart", expanded=True):
        #     hedge_instruments = hedge_df["Hedge Instrument"].tolist()
        #     st.markdown("**Select a Hedge Instrument to view details and charts:**")
        #     selected_hedge_instrument = st.selectbox("", hedge_instruments)



else:
    st.warning("Please upload your basket CSV file or run KDB query to get started.")

st.markdown("---")
st.caption("⚡ MAAS Execution Analytics")
