import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 3 years of synthetic volume data
months = np.tile(np.arange(1, 13), 3)
years = np.repeat([2022, 2023, 2024], 12)
time_index = np.arange(len(months))

# Simulate base volumes (low in summer + Dec)
base_volumes = np.array([110, 115, 120, 125, 120, 90, 85, 90, 115, 120, 115, 95])
volumes = np.tile(base_volumes, 3) + time_index * 0.3 + np.random.normal(0, 3, size=36)

df = pd.DataFrame({'Year': years, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: One-hot encode the Month
df = pd.get_dummies(df, columns=['Month'], prefix='M', drop_first=True)

# Step 3: Train model
feature_cols = [col for col in df.columns if col.startswith('M_')] + ['Time']
X = df[feature_cols]
y = df['Volume']

model = LinearRegression()
model.fit(X, y)

# Step 4: Predict next 12 months
future_months = np.arange(1, 13)
future_year = np.repeat(2025, 12)
future_time_index = np.arange(len(df), len(df) + 12)

future_df = pd.DataFrame({
    'Year': future_year,
    'Month': future_months,
    'Time': future_time_index
})
future_df = pd.get_dummies(future_df, columns=['Month'], prefix='M', drop_first=True)

# Make sure all month columns match training data
for col in feature_cols:
    if col not in future_df.columns:
        future_df[col] = 0

# Predict
future_df['Predicted_Volume'] = model.predict(future_df[feature_cols])

# Step 5: Combine and plot
df['Predicted_Volume'] = model.predict(X)
combined_df = pd.concat([df[['Volume', 'Predicted_Volume']], future_df[['Predicted_Volume']]], ignore_index=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(combined_df.index, combined_df['Volume'], marker='o', label='Actual Volume')
plt.plot(combined_df.index, combined_df['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume')
plt.axvline(x=35.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Equity Market Volume: Actual + Forecast (Month as One-Hot)')
plt.xlabel('Month Index')
plt.ylabel('Avg Volume (Millions)')
plt.legend()
plt.grid(True)
plt.show()
