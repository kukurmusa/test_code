import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(page_title="Hedge Recommendation System", layout="wide", page_icon="🛡️")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = pd.DataFrame()
if 'futures_data' not in st.session_state:
    st.session_state.futures_data = pd.DataFrame()


# Sample data for demonstration
@st.cache_data
def load_sample_data():
    # Sample portfolio assets
    portfolio_assets = {
        'Asset': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'GLD', 'TLT'],
        'Quantity': [1000, 500, 800, 200, 1500, 1000, 300, 500],
        'Current_Price': [185.5, 142.8, 378.2, 248.7, 445.3, 388.9, 198.5, 92.4],
        'Beta': [1.2, 1.1, 0.9, 2.1, 1.0, 1.15, -0.1, -0.8],
        'Volatility': [0.28, 0.25, 0.24, 0.65, 0.18, 0.22, 0.16, 0.12],
        'Correlation_SPY': [0.75, 0.72, 0.68, 0.65, 1.0, 0.85, -0.15, -0.45]
    }

    # Sample futures contracts
    futures_contracts = {
        'Contract': ['ES (S&P 500)', 'NQ (NASDAQ)', 'YM (Dow Jones)', 'RTY (Russell 2000)',
                     'GC (Gold)', 'SI (Silver)', 'CL (Crude Oil)', 'ZN (10Y Treasury)'],
        'Symbol': ['ES', 'NQ', 'YM', 'RTY', 'GC', 'SI', 'CL', 'ZN'],
        'Current_Price': [4485.50, 15632.75, 34567.0, 2045.60, 1985.40, 25.78, 78.45, 112.18],
        'Contract_Size': [50, 20, 5, 50, 100, 5000, 1000, 1000],
        'Margin_Req': [12500, 15000, 8500, 4500, 8800, 16500, 4200, 1800],
        'Correlation_SPY': [0.98, 0.85, 0.92, 0.78, -0.12, 0.05, 0.35, -0.58]
    }

    return pd.DataFrame(portfolio_assets), pd.DataFrame(futures_contracts)


# Main header
st.markdown('<h1 class="main-header">🛡️ Hedge Recommendation System</h1>', unsafe_allow_html=True)

# Load sample data
sample_portfolio, sample_futures = load_sample_data()

# Sidebar for portfolio management
st.sidebar.header("📊 Portfolio Management")

# Portfolio input section
with st.sidebar.expander("Add/Edit Portfolio Assets", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        asset_name = st.text_input("Asset Symbol", value="")
        quantity = st.number_input("Quantity", min_value=0, value=100)
        current_price = st.number_input("Current Price", min_value=0.0, value=100.0)

    with col2:
        beta = st.number_input("Beta", value=1.0, step=0.1)
        volatility = st.number_input("Volatility", min_value=0.0, max_value=2.0, value=0.25, step=0.01)
        correlation = st.number_input("Correlation with SPY", min_value=-1.0, max_value=1.0, value=0.5, step=0.01)

    if st.button("Add Asset"):
        if asset_name:
            new_asset = pd.DataFrame({
                'Asset': [asset_name],
                'Quantity': [quantity],
                'Current_Price': [current_price],
                'Beta': [beta],
                'Volatility': [volatility],
                'Correlation_SPY': [correlation]
            })
            st.session_state.portfolio_data = pd.concat([st.session_state.portfolio_data, new_asset], ignore_index=True)
            st.success(f"Added {asset_name} to portfolio!")

# Load sample data button
if st.sidebar.button("Load Sample Portfolio"):
    st.session_state.portfolio_data = sample_portfolio.copy()
    st.session_state.futures_data = sample_futures.copy()
    st.success("Sample data loaded!")

# Use either session state data or sample data
portfolio_df = st.session_state.portfolio_data if not st.session_state.portfolio_data.empty else sample_portfolio
futures_df = st.session_state.futures_data if not st.session_state.futures_data.empty else sample_futures

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Current Portfolio")

    if not portfolio_df.empty:
        # Calculate portfolio metrics
        portfolio_df['Market_Value'] = portfolio_df['Quantity'] * portfolio_df['Current_Price']
        portfolio_df['Weight'] = portfolio_df['Market_Value'] / portfolio_df['Market_Value'].sum()

        # Display portfolio
        st.dataframe(portfolio_df.style.format({
            'Current_Price': '${:.2f}',
            'Market_Value': '${:,.0f}',
            'Weight': '{:.1%}',
            'Beta': '{:.2f}',
            'Volatility': '{:.1%}',
            'Correlation_SPY': '{:.2f}'
        }), use_container_width=True)

        # Portfolio summary metrics
        total_value = portfolio_df['Market_Value'].sum()
        portfolio_beta = (portfolio_df['Beta'] * portfolio_df['Weight']).sum()
        portfolio_volatility = np.sqrt((portfolio_df['Volatility'] ** 2 * portfolio_df['Weight'] ** 2).sum())

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Portfolio Value", f"${total_value:,.0f}")
        col_b.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        col_c.metric("Portfolio Volatility", f"{portfolio_volatility:.1%}")

with col2:
    st.header("🎯 Risk Profile")

    if not portfolio_df.empty:
        # Risk tolerance settings
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        hedge_ratio = st.slider("Target Hedge Ratio", 0.0, 1.0, 0.8, 0.1)
        time_horizon = st.selectbox("Time Horizon", ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])

        # Market outlook
        market_outlook = st.radio("Market Outlook", ["Bullish", "Neutral", "Bearish"])

# Hedge recommendations section
st.header("🛡️ Hedge Recommendations")

if not portfolio_df.empty:
    # Calculate hedge recommendations
    def calculate_hedge_recommendations(portfolio_df, futures_df, hedge_ratio, risk_tolerance):
        total_portfolio_value = portfolio_df['Market_Value'].sum()
        portfolio_beta = (portfolio_df['Beta'] * portfolio_df['Weight']).sum()

        recommendations = []

        # Strategy 1: Beta-neutral hedge
        for idx, future in futures_df.iterrows():
            if future['Correlation_SPY'] > 0.7:  # High correlation with market
                hedge_value = total_portfolio_value * hedge_ratio
                contracts_needed = hedge_value / (future['Current_Price'] * future['Contract_Size'])

                recommendations.append({
                    'Strategy': 'Beta-Neutral Hedge',
                    'Instrument': future['Contract'],
                    'Symbol': future['Symbol'],
                    'Action': 'Short',
                    'Contracts': int(contracts_needed),
                    'Hedge_Value': hedge_value,
                    'Margin_Required': int(contracts_needed) * future['Margin_Req'],
                    'Effectiveness': abs(future['Correlation_SPY']) * 100,
                    'Risk_Reduction': hedge_ratio * 100
                })

        # Strategy 2: Sector-specific hedge
        tech_weight = portfolio_df[portfolio_df['Asset'].isin(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])]['Weight'].sum()
        if tech_weight > 0.3:  # Significant tech exposure
            nq_future = futures_df[futures_df['Symbol'] == 'NQ'].iloc[0]
            tech_value = total_portfolio_value * tech_weight
            nq_contracts = tech_value * hedge_ratio / (nq_future['Current_Price'] * nq_future['Contract_Size'])

            recommendations.append({
                'Strategy': 'Tech Sector Hedge',
                'Instrument': nq_future['Contract'],
                'Symbol': nq_future['Symbol'],
                'Action': 'Short',
                'Contracts': int(nq_contracts),
                'Hedge_Value': tech_value * hedge_ratio,
                'Margin_Required': int(nq_contracts) * nq_future['Margin_Req'],
                'Effectiveness': 85,
                'Risk_Reduction': hedge_ratio * tech_weight * 100
            })

        # Strategy 3: Volatility hedge
        high_vol_assets = portfolio_df[portfolio_df['Volatility'] > 0.3]
        if not high_vol_assets.empty:
            vix_hedge_value = high_vol_assets['Market_Value'].sum() * 0.1  # 10% of high-vol assets

            recommendations.append({
                'Strategy': 'Volatility Protection',
                'Instrument': 'VIX Futures (Proxy)',
                'Symbol': 'VX',
                'Action': 'Long',
                'Contracts': int(vix_hedge_value / 50000),  # Approximate VIX contract value
                'Hedge_Value': vix_hedge_value,
                'Margin_Required': int(vix_hedge_value / 50000) * 25000,
                'Effectiveness': 70,
                'Risk_Reduction': 15
            })

        return pd.DataFrame(recommendations)


    recommendations_df = calculate_hedge_recommendations(portfolio_df, futures_df, hedge_ratio, risk_tolerance)

    # Display recommendations
    if not recommendations_df.empty:
        for idx, rec in recommendations_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>🎯 {rec['Strategy']}</h4>
                    <p><strong>Instrument:</strong> {rec['Instrument']} ({rec['Symbol']})</p>
                    <p><strong>Action:</strong> {rec['Action']} {rec['Contracts']} contracts</p>
                    <p><strong>Hedge Value:</strong> ${rec['Hedge_Value']:,.0f}</p>
                    <p><strong>Margin Required:</strong> ${rec['Margin_Required']:,.0f}</p>
                    <p><strong>Expected Risk Reduction:</strong> {rec['Risk_Reduction']:.1f}%</p>
                    <p><strong>Hedge Effectiveness:</strong> {rec['Effectiveness']:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)

        # Summary comparison chart
        st.subheader("📊 Hedge Strategy Comparison")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Reduction', 'Hedge Effectiveness', 'Margin Requirements', 'Hedge Value'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Risk reduction chart
        fig.add_trace(
            go.Bar(x=recommendations_df['Strategy'], y=recommendations_df['Risk_Reduction'],
                   name='Risk Reduction %', marker_color='green'),
            row=1, col=1
        )

        # Effectiveness chart
        fig.add_trace(
            go.Bar(x=recommendations_df['Strategy'], y=recommendations_df['Effectiveness'],
                   name='Effectiveness %', marker_color='blue'),
            row=1, col=2
        )

        # Margin requirements
        fig.add_trace(
            go.Bar(x=recommendations_df['Strategy'], y=recommendations_df['Margin_Required'],
                   name='Margin Required', marker_color='orange'),
            row=2, col=1
        )

        # Hedge value
        fig.add_trace(
            go.Bar(x=recommendations_df['Strategy'], y=recommendations_df['Hedge_Value'],
                   name='Hedge Value', marker_color='purple'),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False, title_text="Hedge Strategy Analysis")
        st.plotly_chart(fig, use_container_width=True)

        # Risk-return analysis
        st.subheader("📈 Risk-Return Analysis")

        # Simulate portfolio performance with and without hedge
        np.random.seed(42)
        scenarios = np.random.normal(0, 0.02, 1000)  # Daily returns

        unhedged_returns = scenarios * portfolio_beta
        hedged_returns = unhedged_returns * (1 - hedge_ratio)

        fig_risk = go.Figure()
        fig_risk.add_trace(go.Histogram(x=unhedged_returns, name='Unhedged Portfolio', opacity=0.7))
        fig_risk.add_trace(go.Histogram(x=hedged_returns, name='Hedged Portfolio', opacity=0.7))
        fig_risk.update_layout(
            title='Simulated Daily Returns Distribution',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            barmode='overlay'
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Performance metrics
        col_perf1, col_perf2 = st.columns(2)

        with col_perf1:
            st.metric("Unhedged Volatility", f"{np.std(unhedged_returns):.2%}")
            st.metric("Unhedged VaR (95%)", f"{np.percentile(unhedged_returns, 5):.2%}")

        with col_perf2:
            st.metric("Hedged Volatility", f"{np.std(hedged_returns):.2%}")
            st.metric("Hedged VaR (95%)", f"{np.percentile(hedged_returns, 5):.2%}")

    else:
        st.info("No suitable hedge recommendations found for current portfolio composition.")

# Market data section
st.header("📊 Available Futures Contracts")
st.dataframe(futures_df.style.format({
    'Current_Price': '${:,.2f}',
    'Contract_Size': '{:,.0f}',
    'Margin_Req': '${:,.0f}',
    'Correlation_SPY': '{:.2f}'
}), use_container_width=True)

# Alerts and warnings
st.header("⚠️ Risk Alerts")

if not portfolio_df.empty:
    # Check for concentration risk
    max_weight = portfolio_df['Weight'].max()
    if max_weight > 0.2:
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ Concentration Risk:</strong> 
            {portfolio_df.loc[portfolio_df['Weight'].idxmax(), 'Asset']} represents {max_weight:.1%} of your portfolio.
            Consider additional diversification or specific hedging for this position.
        </div>
        """, unsafe_allow_html=True)

    # Check for high beta exposure
    if portfolio_beta > 1.5:
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ High Beta Risk:</strong> 
            Portfolio beta of {portfolio_beta:.2f} indicates high market sensitivity.
            Consider increasing hedge ratio or using multiple hedge instruments.
        </div>
        """, unsafe_allow_html=True)

    # Check for margin requirements
    if not recommendations_df.empty:
        total_margin = recommendations_df['Margin_Required'].sum()
        if total_margin > total_value * 0.1:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ High Margin Requirements:</strong> 
                Total margin of ${total_margin:,.0f} represents {total_margin / total_value:.1%} of portfolio value.
                Ensure adequate liquidity for margin calls.
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "*This application is for educational purposes only. Please consult with a financial advisor before implementing any hedging strategies.*")
