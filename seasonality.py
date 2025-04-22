import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 1 year of synthetic monthly volume data (2024)
months = np.arange(1, 13)
year = np.repeat(2024, 12)
time_index = np.arange(12)

# Manually define seasonal volumes: low in Jun–Aug, Dec; high in spring/autumn
volume_map = {
    1: 95, 2: 100, 3: 110, 4: 115, 5: 110,
    6: 85, 7: 80, 8: 85,
    9: 110, 10: 115, 11: 110, 12: 90
}
volumes = np.array([volume_map[m] for m in months]) + np.random.normal(0, 2, size=12)

df = pd.DataFrame({'Year': year, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: One-hot encode the Month
df = pd.get_dummies(df, columns=['Month'], prefix='M', drop_first=True)

# Step 3: Train Linear Regression
feature_cols = [col for col in df.columns if col.startswith('M_')] + ['Time']
X = df[feature_cols]
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
future_df = pd.get_dummies(future_df, columns=['Month'], prefix='M', drop_first=True)

# Ensure all expected columns are present
for col in feature_cols:
    if col not in future_df.columns:
        future_df[col] = 0

# Predict volumes
future_df['Predicted_Volume'] = model.predict(future_df[feature_cols])
future_df['Volume'] = np.nan

# Step 5: Combine and label for plotting
combined = pd.concat([df, future_df], ignore_index=True)
combined['Date_Label'] = combined['Year'].astype(str) + '-' + combined.filter(like='M_').idxmax(axis=1).str.extract('(\d+)')[0].fillna('01')
combined['Date_Label'] = pd.to_datetime(combined['Date_Label'], format='%Y-%m').dt.strftime('%b-%Y')

# Step 6: Plot
plt.figure(figsize=(14, 6))
plt.plot(combined.index, combined['Volume'], marker='o', label='Actual Volume (2024)')
plt.plot(combined.index, combined['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume (2025)')
plt.axvline(x=11.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Equity Market Volume: One-Hot Month Model (Lows in Summer & December)')
plt.xlabel('Month')
plt.ylabel('Avg Volume (Millions)')
plt.xticks(ticks=combined.index, labels=combined['Date_Label'], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
