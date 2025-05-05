# 📊 Intraday Reversal Strategy Optimisation (SPY & QQQ)

This project implements and optimises a **mean-reversion intraday trading strategy** using `kdb+/q` and `Python`. The strategy detects sharp intraday drops (>1%) in the **S&P 500 (SPY)** or **Nasdaq 100 (QQQ)** by 1pm EST, then enters a long trade and manages the position with stop-loss, take-profit, and trailing-stop logic.

---

## 📁 Project Structure

```
intraday_reversal/
├── backtest.q              # Core q backtest logic
├── optimise.q              # Grid search optimiser across parameters
├── metrics.q               # Strategy performance metrics (Sharpe, win rate)
├── export_results.q        # Save CSVs for Python visualisation
├── visualise.py            # Python plots and analytics
├── strategy_summary.csv    # Combined optimiser output
├── spy_best_params.csv     # Top 100 SPY results by Sharpe
├── qqq_best_params.csv     # Top 100 QQQ results by Sharpe
├── README.md               # Full documentation
```

---

## ⚙️ Strategy Rules

- **Entry**: If SPY/QQQ drops more than a given threshold from the previous close by 1:00pm EST and price velocity slows.
- **Exit Conditions**:
  - **Stop-Loss**: Exit early if price falls > threshold from entry.
  - **Take-Profit**: Exit early if price rises > threshold from entry.
  - **Trailing Stop**: Exit if price drops > trailing % from max after entry.
  - **Scheduled Exit**: Close at 3:59pm if none of the above are triggered.

---

## 🧠 KDB+/q Components

### `backtest.q`

Implements trade logic, stop-loss, take-profit, trailing stop, and logs:

```q
backtest:{
  sym:x`sym;
  dropThresh:x`drop;
  exitTime:x`exitTime;
  stopLoss:x`stopLoss;
  takeProfit:x`takeProfit;
  trailLoss:x`trailLoss;

  ... / Performs logic
  (`results`skipped)!((flip results); flip skipped)
}
```

### `metrics.q`

```q
computeStats:{
  tbl:x;
  totalPnL: sum tbl`pnl;
  avgPnL: avg tbl`pnl;
  winRate: avg tbl`pnl > 0;
  sharpe: avg tbl`pnl % dev tbl`pnl;
  count: count tbl;
  `count`totalPnL`avgPnL`winRate`sharpe! (count; totalPnL; avgPnL; winRate; sharpe)
}
```

---

## 🔁 Parameter Optimisation (q)

### `optimise.q`

Loops through grid of parameters for both SPY and QQQ:

```q
syms:`SPY`QQQ;
dropVals:-0.012 -0.01 -0.008;
stopLossVals:-0.007 -0.005 -0.003;
takeProfitVals:0.005 0.006 0.007;
trailLossVals:0.003 0.004 0.005;

optResults:();

... / Loop over combinations
optTable: flip optResults;
```

---

## 📤 CSV Export (q)

### `export_results.q`

```q
`:combined_optimisation.csv 0: optTable;
`:spy_best_params.csv 0: top[100] spyBest;
`:qqq_best_params.csv 0: top[100] qqqBest;
```

---

## 📊 Python Visualisation

### `visualise.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("combined_optimisation.csv")
sns.scatterplot(data=df, x="sharpe", y="totalPnL", hue="sym")
plt.title("Sharpe vs Total PnL (SPY vs QQQ)")
```

---

## ✅ Outputs

- `combined_optimisation.csv`: all results (SPY + QQQ)
- `spy_best_params.csv`: top 100 SPY configs
- `qqq_best_params.csv`: top 100 QQQ configs

---

## 🧩 Next Steps

- Add in-sample vs out-of-sample testing
- Explore walk-forward optimisation
- Build dashboard in Dash or Streamlit

---
