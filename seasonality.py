mport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create synthetic monthly volume data for 3 years
months = np.tile(np.arange(1, 13), 3)
years = np.repeat([2022, 2023, 2024], 12)
time_index = np.arange(len(months))

# Lower volumes in summer + Dec
base_volumes = np.array([110, 115, 120, 125, 120, 90, 85, 90, 115, 120, 115, 95])
volumes = np.tile(base_volumes, 3) + time_index * 0.3 + np.random.normal(0, 3, size=36)

df = pd.DataFrame({'Year': years, 'Month': months, 'Volume': volumes, 'Time': time_index})

# Step 2: Train Linear Regression using just Month + Time
X = df[['Month', 'Time']]
y = df['Volume']

model = LinearRegression()
model.fit(X, y)

# Step 3: Predict next 12 months
future_months = np.arange(1, 13)
future_years = np.repeat(2025, 12)
future_time_index = np.arange(len(df), len(df) + 12)

future_df = pd.DataFrame({
    'Year': future_years,
    'Month': future_months,
    'Time': future_time_index
})
future_df['Predicted_Volume'] = model.predict(future_df[['Month', 'Time']])

# Step 4: Combine and plot
df['Predicted_Volume'] = model.predict(X)
combined_df = pd.concat([df[['Volume', 'Predicted_Volume']], future_df[['Predicted_Volume']]], ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(combined_df.index, combined_df['Volume'], marker='o', label='Actual Volume')
plt.plot(combined_df.index, combined_df['Predicted_Volume'], linestyle='--', marker='x', label='Predicted Volume')
plt.axvline(x=35.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Equity Market Volume: Simple Model (Month as Number)')
plt.xlabel('Month Index')
plt.ylabel('Avg Volume (Millions)')
plt.legend()
plt.grid(True)
plt.show()
