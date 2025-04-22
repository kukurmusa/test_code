import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 1 year of historical data (2024)
months = np.arange(1, 13)
year = np.repeat(2024, 12)
time_index = np.arange(12)

# Simulated volume pattern: lows in summer & winter, highs in spring & autumn
base_volume = 100
amplitude = 20
seasonal_volumes = base_volume + amplitude * np.cos(4 * np.pi * (months - 1) / 12)
volumes = seasonal_volumes + np.random.normal(0, 2, size=12)  # add a bit of noise

df = pd.DataFrame({'Year': year, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: Add sine and cosine features
df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)

# Step 3: Train Linear Regression
X = df[['sin_month', 'cos_month', 'Time']]
y = df['Volume']
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict next 12 months (2025)
future_months = np.arange(1, 13)
future_year = np.repeat(2025, 12)
future_time_index = np.arange(len(df), len(df) + 12)

future_df = pd.DataFrame({
    'Year': future_year,
    'Month': future_months,
    'Time': future_time_index
})
future_df['sin_month'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['cos_month'] = np.cos(2 * np.pi * future_df['Month'] / 12)
future_df['Volume'] = np.nan  # unknown actuals
future_df['Predicted_Volume'] = model.predict(future_df[['sin_month', 'cos_month', 'Time']])

# Step 5: Combine and format labels
combined = pd.concat([df, future_df], ignore_index=True)
combined['Date_Label'] = combined['Month'].apply(lambda m: f'{m:02d}') + '-' + combined['Year'].astype(str)

# Step 6: Plot
plt.figure(figsize=(14, 6))
plt.plot(combined.index, combined['Volume'], marker='o', label='Actual Volume (2024)')
plt.plot(combined.index, combined['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume (2025)')
plt.axvline(x=11.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Equity Market Volume Seasonality: Lows in Summer & Winter')
plt.xlabel('Month')
plt.ylabel('Avg Volume (Millions)')
plt.xticks(ticks=combined.index, labels=combined['Date_Label'], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
