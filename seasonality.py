import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create 3 years of synthetic monthly data
months = np.tile(np.arange(1, 13), 3)  # Repeat 1–12 for 3 years
years = np.repeat([2022, 2023, 2024], 12)
time_index = np.arange(len(months))

# Create seasonal sales with some randomness
base_sales = np.array([50, 60, 80, 120, 150, 200, 220, 210, 160, 100, 70, 55])
sales = np.tile(base_sales, 3) + np.random.normal(0, 10, size=36)  # add noise

df = pd.DataFrame({'Year': years, 'Month': months, 'Sales': sales, 'Time': time_index})

# Step 2: Add cyclical features for seasonality
df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)

# Step 3: Train Linear Regression
X = df[['sin_month', 'cos_month', 'Time']]
y = df['Sales']
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

future_df['Predicted_Sales'] = model.predict(future_df[['sin_month', 'cos_month', 'Time']])

# Combine for plotting
df['Predicted_Sales'] = model.predict(X)
combined_df = pd.concat([df, future_df], ignore_index=True)

# Step 5: Plot actual + predicted
plt.figure(figsize=(12, 6))
plt.plot(combined_df.index, combined_df['Sales'], marker='o', label='Actual Sales')
plt.plot(combined_df.index, combined_df['Predicted_Sales'], linestyle='--', marker='x', label='Predicted Sales')
plt.axvline(x=35.5, color='grey', linestyle=':', label='Forecast Starts (2025)')
plt.title('Ice Cream Sales: Actual (3 years) + Forecast (1 year)')
plt.xlabel('Month Index')
plt.ylabel('Sales (£)')
plt.legend()
plt.grid(True)
plt.show()
