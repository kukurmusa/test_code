import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example data
np.random.seed(1)
df = pd.DataFrame({
    'x': np.tile(np.arange(10), 3),
    'y': np.random.rand(30).cumsum(),
    'group': np.repeat(['A', 'B', 'C'], 10)
})

# Plot using seaborn to get the colours
palette = sns.color_palette("husl", n_colors=df['group'].nunique())

plt.figure(figsize=(8, 5))

for i, (name, group_df) in enumerate(df.groupby('group')):
    plt.plot(group_df['x'], group_df['y'], label=name, color=palette[i])
    plt.fill_between(group_df['x'], group_df['y'], alpha=0.3, color=palette[i])

plt.legend(title='Group')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot with Filled Areas by Group')
plt.tight_layout()
plt.show()