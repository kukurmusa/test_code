import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
df = sns.load_dataset("tips").rename(columns={"total_bill": "value"})
# (Optional) make it more dispersed for demo:
# df.loc[df.sample(frac=0.05, random_state=1).index, "value"] *= 3

# ---------- OPTION A: GLOBAL ±4σ CLIP ----------
n_sigma = 4
mu, sd = df["value"].mean(), df["value"].std()
df_clip_global = df.loc[df["value"].between(mu - n_sigma*sd, mu + n_sigma*sd)].copy()

g = sns.catplot(
    data=df_clip_global,
    x="day", y="value", hue="sex",
    kind="violin",
    inner="quartile",        # show quartiles inside the violin
    cut=0,                   # don't extend KDE beyond the data range
    scale="width",           # helpful when groups are imbalanced
    bw_adjust=0.8,           # slightly smoother KDE
    common_norm=False        # keeps each hue/group on its own scale
)
g.set_axis_labels("Day", "Value")
g.fig.suptitle("Violin (Global clip at ±4σ)", y=1.02)
plt.show()

# ---------- OPTION B: PER-GROUP ±4σ CLIP ----------
# Useful when dispersion differs a lot by category/hue.
group_cols = ["day", "sex"]  # match your x/hue grouping
def clip_group(s, n=4):
    m, st = s.mean(), s.std(ddof=1)
    return s[(s >= m - n*st) & (s <= m + n*st)]

df_clip_group = (
    df.groupby(group_cols, group_keys=False)
      .apply(lambda g: g.assign(value=clip_group(g["value"], n_sigma)))
      .dropna(subset=["value"])
)

g2 = sns.catplot(
    data=df_clip_group,
    x="day", y="value", hue="sex",
    kind="violin",
    inner="quartile",
    cut=0,
    scale="width",
    bw_adjust=0.8,
    common_norm=False
)
g2.set_axis_labels("Day", "Value")
g2.fig.suptitle("Violin (Per-group clip at ±4σ)", y=1.02)
plt.show()
