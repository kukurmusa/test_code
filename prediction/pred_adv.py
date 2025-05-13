import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import boxcox

# Load main dataset (includes Close, Volume, VIX, SPX_Close)
df = pd.read_csv("stock_data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# --- Simulated external sources ---
adr_df = pd.read_csv("adr_data.csv", parse_dates=["Date"])  # ADR_Close, ADR_Volume, ADR_Return
futures_df = pd.read_csv("futures_data.csv", parse_dates=["Date"])  # ESTX50_Fut_Volume, ESTX50_Fut_Return

# Merge external features
df = df.merge(adr_df, on="Date", how="left")
df = df.merge(futures_df, on="Date", how="left")

# SPX correlation (10-day rolling)
df['SPX_Correlation'] = df['Close'].rolling(window=10).corr(df['SPX_Close'])

# --- Feature engineering ---
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(window=5).std()
df['Volume_Lag1'] = df['Volume'].shift(1)
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_10'] = df['Close'].ewm(span=10).mean()
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month

# Lagged ADR and Futures return
df['ADR_Return_Lag1'] = df['ADR_Return'].shift(1)
df['Fut_Return_Lag1'] = df['ESTX50_Fut_Return'].shift(1)

# Drop rows with NaN (from rolling & merging)
df.dropna(inplace=True)

# --- Box-Cox transform to stabilise target ---
df['Volume'], lam = boxcox(df['Volume'])

# --- Final feature list ---
features = [
    'Return', 'Volatility', 'Volume_Lag1', 'SMA_10', 'EMA_10',
    'VIX', 'SPX_Correlation', 'DayOfWeek', 'Month',
    'ADR_Volume', 'ADR_Return', 'ADR_Return_Lag1',
    'ESTX50_Fut_Volume', 'ESTX50_Fut_Return', 'Fut_Return_Lag1'
]

X = df[features]
y = df['Volume']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- LightGBM Training ---
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

# --- Prediction and Inverse Transform ---
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred) if lam == 0 else np.power(y_pred * lam + 1, 1 / lam)
y_test = np.exp(y_test) if lam == 0 else np.power(y_test * lam + 1, 1 / lam)

# --- Evaluation ---
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# --- Feature Importance ---
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title("Feature Importance - LightGBM Volume Forecast")
plt.show()

# --- Prediction Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test, label="Actual Volume", marker='o')
plt.plot(df.index[-len(y_test):], y_pred, label="Predicted Volume", linestyle='--')
plt.title("Forecasted vs Actual Volume")
plt.legend()
plt.show()
