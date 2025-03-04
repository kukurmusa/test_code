import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame
data = {
    'time': pd.date_range(start='10:00', periods=10, freq='T'),  # Generating time
    'size': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'value1': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    'value2': [200, 180, 160, 140, 120, 100, 80, 60, 40, 20]
}
df = pd.DataFrame(data)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# First Y-axis
sns.lineplot(x=df['time'], y=df['size'], ax=ax1, color='b', label='Size')
ax1.set_ylabel('Size', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Second Y-axis
ax2 = ax1.twinx()
sns.lineplot(x=df['time'], y=df['value1'], ax=ax2, color='r', label='Value1')
ax2.set_ylabel('Value1', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Third Y-axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
sns.lineplot(x=df['time'], y=df['value2'], ax=ax3, color='g', label='Value2')
ax3.set_ylabel('Value2', color='g')
ax3.tick_params(axis='y', labelcolor='g')

# Formatting
ax1.set_xlabel('Time')
ax1.set_xticklabels(df['time'].dt.strftime('%H:%M'), rotation=45)
fig.tight_layou
