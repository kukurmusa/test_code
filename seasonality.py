import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 3 years of synthetic volume data
months = np.tile(np.arange(1, 13), 3)
years = np.repeat([2022, 2023, 2024], 12)
time_index = np.arange(len(months))

# Create base volume levels with expected seasonal dips
base_volumes = np.array([100, 105, 110, 115, 110, 90, 85, 90, 110, 115, 110, 95])
# Note: dips in June–Aug and Dec

# Add a slight upward trend and noise
volumes = np.tile(base_volumes, 3) + time_index * 0.5 + np.random.normal(0, 3, size=36)

df = pd.DataFrame({'Year': years, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: Feature engineering (seasonal + time trend)
df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)

# Step 3: Train linear regression model
X = df[['sin_month', 'cos_month', 'Time']]
y = df['Volume']

model = LinearRegression()
model.fit(X, y)

# Step 4: Predict future (12 months ahead = 2025)
future_months = np.arange(1, 13)
future_years = np.repeat(2025, 12)
future_time_index = np.arange(len(df), len(df) + 12)

future_df = pd.DataFrame({
    'Year': future_years,
    'Month': future_months,
    'Time': future_time_index
})
future_df['sin_month'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['cos_month'] = np.cos(2 * np.pi * future_df['Month'] / 12)

future_df['Predicted_Volume'] = model.predict(future_df[['sin_month', 'cos_month', 'Time']])

# Combine actual and future for plotting
df['Predicted_Volume'] = model.predict(X)
combined_df = pd.concat([df, future_df], ignore_index=True)

# Step 5: Plot actual vs predicted volumes
plt.figure(figsize=(12, 6))
plt.plot(combined_df.index, combined_df['Volume'], marker='o', label='Actual Volume')
plt.plot(combined_df.index, combined_df['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume')
plt.axvline(x=35.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Monthly Average Equity Market Volume (with Seasonality)')
plt.xlabel('Month Index')
plt.ylabel('Avg Volume (Millions)')
plt.legend()
plt.grid(True)
plt.show()
