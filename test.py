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

# First Y-axis (Primary)
sns.lineplot(x=df['time'], y=df['size'], ax=ax1, color='b', label='Size')
ax1.set_ylabel('Size', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Second Y-axis
ax2 = ax1.twinx()
line1, = ax2.plot(df['time'], df['value1'], color='r', label='Value1')  # Save handle for legend
ax2.set_ylabel('Value1', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Third Y-axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset third axis
line2, = ax3.plot(df['time'], df['value2'], color='g', label='Value2')  # Save handle for legend
ax3.set_ylabel('Value2', color='g')
ax3.tick_params(axis='y', labelcolor='g')

# Combining Legends
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.1, 1))

# Formatting
ax1.set_xlabel('Time')
ax1.set_xticklabels(df['time'].dt.strftime('%H:%M'), rotation=45)
fig.tight_layout()
plt.show()
