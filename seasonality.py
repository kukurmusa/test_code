import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 3 years of synthetic monthly data
months = np.tile(np.arange(1, 13), 3)
years = np.repeat([2022, 2023, 2024], 12)
time_index = np.arange(len(months))

# Simulated seasonal pattern: low in summer + winter, high in spring/autumn
seasonal_pattern = 100 + 20 * np.cos(2 * np.pi * (months - 3) / 12)  # peak around March/September

# Add a gentle upward trend and noise
volumes = seasonal_pattern + time_index * 0.4 + np.random.normal(0, 3, size=36)

df = pd.DataFrame({'Year': years, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: Add sine and cosine features for seasonality
df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)

# Step 3: Train Linear Regression
X = df[['sin_month', 'cos_month', 'Time']]
y = df['Volume']
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict next 12 months (2025)
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
future_df['Volume'] = np.nan  # mark as unknown

# Step 5: Combine and create labels
combined = pd.concat([df, future_df], ignore_index=True)
combined['Date_Label'] = combined['Month'].apply(lambda m: f'{m:02d}') + '-' + combined['Year'].astype(str)

# Step 6: Plot
plt.figure(figsize=(14, 6))
plt.plot(combined.index, combined['Volume'], marker='o', label='Actual Volume')
plt.plot(combined.index, combined['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume')
plt.axvline(x=35.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Equity Market Volume with Seasonal Pattern (Low in Summer & Winter)')
plt.xlabel('Month')
plt.ylabel('Avg Volume (Millions)')
plt.xticks(ticks=combined.index, labels=combined['Date_Label'], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
