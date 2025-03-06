import matplotlib.pyplot as plt
import numpy as np

# Convert Timestamp to datetime for proper plotting
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S")

# Convert Overall POV to numeric (removing percentage sign)
df["Overall POV"] = df["Overall POV"].str.rstrip('%').astype(float)

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line chart for Indicative Size
ax1.plot(df["Timestamp"], df["Indicative Size"], marker='o', linestyle='-', label="Indicative Size", color='blue')
ax1.set_ylabel("Indicative Size")
ax1.set_xlabel("Timestamp")

# Secondary Y-axis for stacked area chart
ax2 = ax1.twinx()
ax2.stackplot(
    df["Timestamp"],
    df["Must Complete (10%)"],
    df["Scaling Algo Child Order Size"],
    df["Opportunistic Qty"].fillna(0),
    labels=["Must Complete", "Scaling Algo Child Order", "Opportunistic Qty"],
    alpha=0.5
)
ax2.set_ylabel("Order Quantities")

# Third Y-axis for Overall POV
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(df["Timestamp"], df["Overall POV"], marker='s', linestyle='--', label="Overall POV", color='black')
ax3.set_ylabel("Overall POV (%)")

# Legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Another line chart for indicative and opportunistic prices
fig, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(df["Timestamp"], df["Indicative Price"], marker='o', linestyle='-', label="Indicative Price", color='green')
ax4.plot(df["Timestamp"], df["Opportunistic Price"], marker='s', linestyle='--', label="Opportunistic Price", color='red')
ax4.set_xlabel("Timestamp")
ax4.set_ylabel("Price")
ax4.legend()
plt.xticks(rotation=45)

# Display the plots
plt.show()
