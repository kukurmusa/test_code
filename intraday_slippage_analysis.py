
"""
Intraday Factor Return Estimation and Arrival Slippage Decomposition
====================================================================
This script simulates intraday stock returns, estimates Barra-style factor returns 
using weighted linear regression, and decomposes arrival slippage of sample orders 
into systematic (factor) and residual components.

Author: OpenAI ChatGPT
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
np.random.seed(42)

# ---------------------------
# Simulate intraday data
# ---------------------------
logging.info("Generating synthetic intraday data for stocks...")

symbols = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GOOG']
factors = ['Value', 'Momentum', 'Size']
minutes = pd.date_range('2024-01-02 09:30', '2024-01-02 16:00', freq='1min')

data = []
for t in minutes:
    for sym in symbols:
        returns = np.random.normal(0, 0.0005)
        exposures = np.random.normal(0, 1, len(factors))
        data.append([t, sym, returns, *exposures])

columns = ['datetime', 'symbol', 'return'] + factors
df = pd.DataFrame(data, columns=columns)

# Assign fixed market caps to simulate weights
market_caps = {
    'AAPL': 2.5e12,
    'MSFT': 2.0e12,
    'JPM': 0.5e12,
    'XOM': 0.4e12,
    'GOOG': 1.8e12
}
df['mcap'] = df['symbol'].map(market_caps)
df['weight'] = df['mcap'] / df.groupby('datetime')['mcap'].transform('sum')

# ---------------------------
# Estimate factor returns
# ---------------------------
def compute_weighted_minutely_factor_returns(df, factors):
    logging.info("Estimating factor returns using weighted regression...")
    factor_returns = []
    grouped = df.groupby('datetime')
    for dt, group in grouped:
        if len(group) < len(factors):
            continue
        X = group[factors]
        y = group['return']
        w = group['weight']
        try:
            model = sm.WLS(y, sm.add_constant(X), weights=w).fit()
            returns = model.params.drop('const')
            returns.name = dt
            factor_returns.append(returns)
        except Exception as e:
            logging.warning(f"Regression failed at {dt}: {e}")
    return pd.DataFrame(factor_returns)

factor_return_df = compute_weighted_minutely_factor_returns(df, factors)

# ---------------------------
# Decompose arrival slippage
# ---------------------------
def decompose_arrival_slippage(P_arrival, P_exec, beta, factor_returns, t_arrival, t_exec):
    slippage_bps = ((P_exec - P_arrival) / P_arrival) * 1e4
    f_start = factor_returns.loc[t_arrival]
    f_end = factor_returns.loc[t_exec]
    delta_f = f_end - f_start
    predicted_return = np.dot(beta, delta_f)
    predicted_slippage_bps = predicted_return * 1e4
    residual_bps = slippage_bps - predicted_slippage_bps
    return {
        'slippage_bps': slippage_bps,
        'factor_slippage_bps': predicted_slippage_bps,
        'residual_slippage_bps': residual_bps
    }

# ---------------------------
# Generate and evaluate sample orders
# ---------------------------
logging.info("Generating and decomposing sample order slippages...")

order_count = 20
np.random.seed(123)
symbols_sample = np.random.choice(symbols, order_count)
arrival_times = np.random.choice(factor_return_df.index[:-10], order_count)
exec_times = [t + pd.Timedelta(minutes=np.random.randint(1, 10)) for t in arrival_times]
prices_arrival = np.random.uniform(95, 105, order_count)
prices_exec = prices_arrival + np.random.normal(0, 0.2, order_count)
betas = np.random.normal(0, 1, (order_count, len(factors)))

decompositions = []
for i in range(order_count):
    t_arr = arrival_times[i]
    t_exec = exec_times[i]
    if t_arr not in factor_return_df.index or t_exec not in factor_return_df.index:
        continue
    result = decompose_arrival_slippage(
        P_arrival=prices_arrival[i],
        P_exec=prices_exec[i],
        beta=betas[i],
        factor_returns=factor_return_df,
        t_arrival=t_arr,
        t_exec=t_exec
    )
    result.update({
        'symbol': symbols_sample[i],
        'arrival_time': t_arr,
        'exec_time': t_exec,
        'arrival_price': prices_arrival[i],
        'exec_price': prices_exec[i]
    })
    decompositions.append(result)

orders_df = pd.DataFrame(decompositions)
logging.info("Finished decomposing slippage for sample orders.")

# Save output
orders_df.to_csv("arrival_slippage_decomposition.csv", index=False)
logging.info("Saved results to 'arrival_slippage_decomposition.csv'")
