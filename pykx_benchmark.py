# --- PyKX vs Pandas vs Polars Mini Benchmark ---
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt

# optional imports
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pykx as kx
    HAS_PYKX = True
except ImportError:
    HAS_PYKX = False

# -----------------------------
# Generate test data
# -----------------------------
N = 1_000_000
S = 50
rng = np.random.default_rng(42)
syms = np.array([f"S{i:03d}" for i in range(S)])
ticks = pd.DataFrame({
    "sym": rng.choice(syms, N),
    "px": rng.random(N) * 100 + 10,
    "qty": rng.integers(1, 1000, N),
    "ts": np.arange(N)
})
ref = pd.DataFrame({
    "sym": syms,
    "sector": [f"SEC{i%5}" for i in range(S)],
    "beta": np.linspace(0.8, 1.2, S)
})

# -----------------------------
# Test cases
# -----------------------------
def bench_pandas():
    out = {}
    start = time.time()
    out["groupby_mean"] = ticks.groupby("sym")["px"].mean()
    t1 = time.time() - start

    start = time.time()
    out["vwap"] = ticks.groupby("sym").apply(lambda d: (d.px*d.qty).sum()/d.qty.sum())
    t2 = time.time() - start

    start = time.time()
    m = ticks.merge(ref, on="sym", how="left")
    out["join_group"] = m.groupby("sector")["px"].mean()
    t3 = time.time() - start

    start = time.time()
    df2 = ticks.sort_values(["sym","ts"]).copy()
    df2["roll_px"] = df2.groupby("sym")["px"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    t4 = time.time() - start

    start = time.time()
    out["conditional"] = ticks.loc[ticks.qty>900, "px"].mean()
    t5 = time.time() - start
    return [t1,t2,t3,t4,t5]

def bench_polars():
    pl_t = pl.from_pandas(ticks)
    pl_r = pl.from_pandas(ref)
    out = {}
    start = time.time()
    out["groupby_mean"] = pl_t.groupby("sym").agg(pl.col("px").mean())
    t1 = time.time() - start

    start = time.time()
    out["vwap"] = pl_t.groupby("sym").agg(((pl.col("px")*pl.col("qty")).sum()/pl.col("qty").sum()).alias("vwap"))
    t2 = time.time() - start

    start = time.time()
    out["join_group"] = pl_t.join(pl_r, on="sym", how="left").groupby("sector").agg(pl.col("px").mean())
    t3 = time.time() - start

    start = time.time()
    out["rolling"] = pl_t.sort(["sym","ts"]).with_columns(
        pl.col("px").rolling_mean(window_size=20).over("sym").alias("roll_px")
    )
    t4 = time.time() - start

    start = time.time()
    out["conditional"] = pl_t.filter(pl.col("qty")>900)["px"].mean()
    t5 = time.time() - start
    return [t1,t2,t3,t4,t5]

def bench_pykx():
    t = kx.Table(ticks)
    r = kx.Table(ref)
    q = kx.q
    start = time.time()
    q('select mean_px:avg px by sym from t', t)
    t1 = time.time() - start

    start = time.time()
    q('select vwap:sum px*qty % sum qty by sym from t', t)
    t2 = time.time() - start

    start = time.time()
    q('select mean_px_by_sector:avg px by sector from lj[`sym;t;r]', t, r)
    t3 = time.time() - start

    start = time.time()
    q('{[t] update roll_px:20 mavg px by sym from t}', t)
    t4 = time.time() - start

    start = time.time()
    q('avg px from t where qty>900', t)
    t5 = time.time() - start
    return [t1,t2,t3,t4,t5]

# -----------------------------
# Run benchmarks
# -----------------------------
tests = ["Groupby mean","VWAP","Join+Group","Rolling mean","Conditional mean"]
results = {}

results["pandas"] = bench_pandas()
if HAS_POLARS:
    results["polars"] = bench_polars()
if HAS_PYKX:
    results["pykx"] = bench_pykx()

# -----------------------------
# Display results
# -----------------------------
df_res = pd.DataFrame(results, index=tests)
df_res.loc["Average"] = df_res.mean()
display(df_res.style.format("{:.4f}s").highlight_min(color="lightgreen", axis=1))

# Plot
df_res.drop("Average").plot(kind="bar", figsize=(9,4))
plt.ylabel("Seconds (lower = faster)")
plt.title("PyKX vs Pandas vs Polars benchmark")
plt.xticks(rotation=30)
plt.show()
