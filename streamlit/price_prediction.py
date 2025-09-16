import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ======================== CONFIG (edit defaults here) ========================
CONFIG = {
    "use_sample": True,          # Set False to upload CSVs below
    "default_horizon": 5,        # e.g., 5/10/15
    "show_class_markers": True,  # show markers on the price chart
    "show_up_init": True,        # initial visibility for Up markers
    "show_down_init": True,      # initial visibility for Down markers
    "show_flat_init": False,     # initial visibility for Flat markers (hidden by default)
    "marker_size": 9,
    "marker_opacity": 0.85,
    "bps_bar_width_min": 2,      # width of Δbps bars (minutes)
    "show_volume": True,         # add 1-minute volume on secondary axis (top chart)
    "vol_axis_range_multiplier": 4.0,  # larger => bars look smaller (uses 95th pct * this)
    "vol_opacity": 0.35,         # visual subtlety
    "vol_bar_width_min": 1,      # minute width for volume bars
}

st.set_page_config(page_title="Intraday Price + Δbps + Volume (Select Horizon)", page_icon="📈", layout="wide")
st.title("📈 Intraday Price (top) + Predicted Δbps (bottom) — Select Horizon")

# ----------------------- Sample data helpers --------------------
def make_sample_price(start=None, minutes=390, seed=42):
    rng = np.random.default_rng(seed)
    if start is None:
        start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    ts = pd.date_range(start=start, periods=minutes, freq="1min")
    rets = rng.normal(0, 0.0008, size=minutes)
    price = 100 * (1 + rets).cumprod()
    # Synthetic 1-min volumes with intraday U-shape
    base = rng.lognormal(mean=9.2, sigma=0.5, size=minutes)  # ~thousands
    minute = np.arange(minutes)
    u_shape = 0.7 + 0.6 * (np.cos((minute / minutes) * 2 * np.pi) * -1)  # more vol at open/close
    vol = (base * u_shape).astype(int)
    return pd.DataFrame({"timestamp": ts, "price": price, "volume": vol})

def make_sample_preds(price_df: pd.DataFrame, horizons=(5,10,15), seed=43):
    rng = np.random.default_rng(seed)
    rows = []
    for t in price_df["timestamp"]:
        for h in horizons:
            cls = rng.choice(["Up", "Down", "Flat"], p=[0.35, 0.35, 0.30])
            conf = rng.uniform(0.55, 0.95)
            pred_bps = rng.normal(0, 6)
            rows.append({
                "timestamp": t,
                "horizon_min": h,
                "class": cls,
                "confidence": conf,
                "pred_bps": pred_bps
            })
    return pd.DataFrame(rows)

# ---------------------------- Load data --------------------------
if CONFIG["use_sample"]:
    df_price = make_sample_price()
    df_pred = make_sample_preds(df_price)
else:
    st.subheader("Upload data")
    price_file = st.file_uploader("Price CSV (timestamp, price, volume)", type=["csv"])
    preds_file = st.file_uploader("Predictions CSV (timestamp, horizon_min, class, pred_bps[, confidence])", type=["csv"])
    if not price_file or not preds_file:
        st.stop()
    df_price = pd.read_csv(price_file)
    df_pred = pd.read_csv(preds_file)

def ensure_ts(df, col="timestamp"):
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])

df_price = ensure_ts(df_price, "timestamp").sort_values("timestamp")
for col in ["price", "volume"]:
    if col not in df_price.columns:
        st.error("Price CSV must include 'price' and 'volume' columns (1-minute binned).")
        st.stop()

df_pred = ensure_ts(df_pred, "timestamp").sort_values("timestamp")
needed = {"horizon_min", "class", "pred_bps"}
missing = needed - set(df_pred.columns)
if missing:
    st.error(f"Predictions CSV missing columns: {missing}. Required: timestamp, horizon_min, class, pred_bps[, confidence].")
    st.stop()

# Compute END time & realised END price
df_pred["end_time"] = df_pred["timestamp"] + pd.to_timedelta(df_pred["horizon_min"], unit="m")
df_price_end = df_price.rename(columns={"timestamp": "end_time", "price": "end_price"})
df_pred = df_pred.merge(df_price_end[["end_time", "end_price"]], on="end_time", how="left").dropna(subset=["end_price"])

# --------------------- Horizon picker (main pane) ---------------------
available_horizons = sorted(df_pred["horizon_min"].dropna().unique().tolist())
default_idx = available_horizons.index(CONFIG["default_horizon"]) if CONFIG["default_horizon"] in available_horizons else 0
selected_h = st.selectbox("Prediction horizon (minutes):", available_horizons, index=default_idx, key="horizon_select")

dfh = df_pred[df_pred["horizon_min"] == selected_h].copy()
dfh["class"] = dfh["class"].astype(str).str.capitalize()

# --------------------------- Figure scaffold ---------------------
# Top row has secondary_y for volume; bottom row for Δbps
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.68, 0.32], vertical_spacing=0.06,
    specs=[[{"secondary_y": True}], [{}]]
)

# --------------------------- Top: Price line ---------------------
fig.add_trace(
    go.Scatter(
        x=df_price["timestamp"], y=df_price["price"],
        mode="lines", name="Trade Price",
        hovertemplate="Time: %{x}<br>Price: %{y:.4f}<extra></extra>"
    ),
    row=1, col=1, secondary_y=False
)

# --------------------------- Top: Volume (secondary axis) ---------------------
if CONFIG["show_volume"]:
    vol95 = float(np.nanpercentile(df_price["volume"], 95))
    # Make bars visually small by using a *larger* axis range (multiplier), hide ticks/label.
    vol_axis_max = vol95 * CONFIG["vol_axis_range_multiplier"]

    fig.add_trace(
        go.Bar(
            x=df_price["timestamp"], y=df_price["volume"],
            name="Volume (1m)",
            width=int(CONFIG["vol_bar_width_min"] * 60 * 1000),  # minutes -> ms
            opacity=CONFIG["vol_opacity"],
            marker_line_width=0,
            hovertemplate="Time: %{x}<br>Volume: %{y:,}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True
    )

    fig.update_yaxes(
        title_text="", showgrid=False, showticklabels=False, zeroline=False,
        range=[0, max(vol_axis_max, df_price["volume"].max() * 1.05)],
        secondary_y=True, row=1, col=1
    )

# --------------------------- Top: Classification markers ----------------------
if CONFIG["show_class_markers"] and not dfh.empty:
    class_to_symbol = {"Up": "triangle-up", "Down": "triangle-down", "Flat": "circle"}
    class_init_vis = {"Up": CONFIG["show_up_init"], "Down": CONFIG["show_down_init"], "Flat": CONFIG["show_flat_init"]}

    for cls in ["Up", "Down", "Flat"]:
        dfg = dfh[dfh["class"] == cls]
        if dfg.empty:
            continue
        init_visible = True if class_init_vis.get(cls, True) else "legendonly"

        if "confidence" in dfg.columns:
            custom = np.stack([dfg["timestamp"].dt.strftime("%Y-%m-%d %H:%M"), dfg["confidence"]], axis=1)
            hover = (
                "END Time: %{x}<br>END Price: %{y:.4f}"
                f"<br>Pred Time: %{{customdata[0]}}<br>Horizon: {selected_h}m<br>Class: {cls}"
                "<br>Confidence: %{{customdata[1]:.0%}}<extra></extra>"
            )
        else:
            custom = np.stack([dfg["timestamp"].dt.strftime("%Y-%m-%d %H:%M")], axis=1)
            hover = (
                "END Time: %{x}<br>END Price: %{y:.4f}"
                f"<br>Pred Time: %{{customdata[0]}}<br>Horizon: {selected_h}m<br>Class: {cls}<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=dfg["end_time"], y=dfg["end_price"],
                mode="markers",
                name=f"{selected_h}m {cls}",
                marker=dict(size=CONFIG["marker_size"], symbol=class_to_symbol[cls]),
                opacity=CONFIG["marker_opacity"],
                hovertemplate=hover,
                customdata=custom,
                visible=init_visible,
                legendgroup=f"{selected_h}m {cls}",
            ),
            row=1, col=1, secondary_y=False
        )

# ----------------------- Bottom: Δbps bars (selected horizon) -------------------
if not dfh.empty:
    colours = np.where(dfh["pred_bps"] > 0, "green", np.where(dfh["pred_bps"] < 0, "red", "grey"))

    fig.add_trace(
        go.Bar(
            x=dfh["end_time"], y=dfh["pred_bps"],
            name=f"{selected_h}m Δbps (pred)",
            width=int(CONFIG["bps_bar_width_min"] * 60 * 1000),  # minutes -> ms
            marker_color=colours,
            opacity=0.9,
            hovertemplate=(
                "END Time: %{x}<br>Predicted Δ: %{y:.2f} bps"
                f"<br>Horizon: {selected_h}m"
                + ("<br>Confidence: %{{customdata[0]:.0%}}" if "confidence" in dfh.columns else "")
                + "<extra></extra>"
            ),
            customdata=dfh[["confidence"]] if "confidence" in dfh.columns else None,
        ),
        row=2, col=1
    )

# --------------------------- Layout polish ----------------------
fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Predicted Δ (bps)", row=2, col=1, zeroline=True, zerolinewidth=1)

fig.update_layout(
    height=840,
    margin=dict(l=10, r=10, t=32, b=10),
    hovermode="x unified",
    # NO range sliders anywhere:
    xaxis=dict(rangeslider=dict(visible=False)),
    xaxis2=dict(rangeslider=dict(visible=False)),
    barmode="overlay"
)

# Shared x; show helpful spikes but no sliders
fig.update_xaxes(matches='x', row=1, col=1, showspikes=True, spikemode="across")
fig.update_xaxes(matches='x', row=2, col=1, showspikes=True, spikemode="across")

st.plotly_chart(fig, use_container_width=True)

# --------------------------- Notes ------------------------------
with st.expander("Data format & behaviour"):
    st.markdown("""
**Price CSV**: `timestamp`, `price`, `volume` (1-minute binned)  
**Predictions CSV**: `timestamp` (prediction time), `horizon_min` (int), `class` (Up/Down/Flat), `pred_bps` (float), optional `confidence`.

- Volume is drawn on the **secondary Y-axis** of the **top chart** as subtle bars.  
- Its visual size is controlled by `vol_axis_range_multiplier` (higher → smaller bars).  
- Use the legend to enable/disable **Up/Down/Flat** markers (Flat hidden by default).  
- Both charts are synced; there’s **no range slider**.
""")
