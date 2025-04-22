import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 2 years of data (2023–2024)
months = np.tile(np.arange(1, 13), 2)
years = np.repeat([2023, 2024], 12)
time_index = np.arange(24)

# Define a seasonal volume pattern
volume_map = {
    1: 95, 2: 100, 3: 110, 4: 115, 5: 110,
    6: 85, 7: 80, 8: 85,
    9: 110, 10: 115, 11: 110, 12: 90
}
seasonal_volumes = np.array([volume_map[m] for m in months])
volumes = seasonal_volumes + time_index * 0.3 + np.random.normal(0, 2, size=24)

df = pd.DataFrame({'Year': years, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: One-hot encode the Month
df = pd.get_dummies(df, columns=['Month'], prefix='M', drop_first=True)

# Step 3: Train Linear Regression
feature_cols = [col for col in df.columns if col.startswith('M_')] + ['Time']
X = df[feature_cols]
y = df['Volume']
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict for next 12 months (2025)
future_months = np.arange(1, 13)
future_years = np.repeat(2025, 12)
future_time_index = np.arange(len(df), len(df) + 12)

future_df = pd.DataFrame({
    'Year': future_years,
    'Month': future_months,
    'Time': future_time_index
})
future_df = pd.get_dummies(future_df, columns=['Month'], prefix='M', drop_first=True)

# Ensure all feature columns are present
for col in feature_cols:
    if col not in future_df.columns:
        future_df[col] = 0

future_df['Predicted_Volume'] = model.predict(future_df[feature_cols])
future_df['Volume'] = np.nan

# Step 5: Combine for plotting
combined = pd.concat([df, future_df], ignore_index=True)

# Add Month-Year labels
combined['Month_Num'] = np.tile(np.arange(1, 13), 3)
combined['Date_Label'] = combined['Year'].astype(str) + '-' + combined['Month_Num'].astype(str).str.zfill(2)
combined['Date_Label'] = pd.to_datetime(combined['Date_Label'], format='%Y-%m').dt.strftime('%b-%Y')

# Step 6: Plot
plt.figure(figsize=(14, 6))
plt.plot(combined.index, combined['Volume'], marker='o', label='Actual Volume (2023–2024)')
plt.plot(combined.index, combined['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume (2025)')
plt.axvline(x=23.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Equity Market Volume with Seasonality (One-Hot Month Model)')
plt.xlabel('Month')
plt.ylabel('Avg Volume (Millions)')
plt.xticks(ticks=combined.index, labels=combined['Date_Label'], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
