
# Overnight Equities Volume Forecasting (European Symbols)

## 🧾 Objective
Forecast the *next business day's full-day volume* for each equity symbol in a European universe, using data available up to 1 hour before market open.

---

## 📥 Model Inputs (Features)

| Feature             | Type         | Description |
|---------------------|--------------|-------------|
| `symbol_cat`        | Categorical  | Encoded ID for each stock symbol |
| `prev_close`        | Numerical    | Previous day's close price |
| `log_prev_volume`   | Numerical    | Log-transformed previous day's volume |
| `return`            | Numerical    | Previous day's close-to-close return |
| `vix_change`        | Numerical    | Change in VIX index overnight |
| `stoxx_fut_chg`     | Numerical    | % change in STOXX futures from prior close to current overnight |
| `dow_chg`           | Numerical    | % return of the Dow Jones index from previous close |
| `fx_eurusd_chg`     | Numerical    | % change in EUR/USD FX rate overnight |
| `dow_vol_adr`       | Numerical    | Volume multiplier based on US ADR activity for dual-listed stocks |
| `is_month_end`      | Binary       | Flag if the date is a month-end (1) or not (0) |

---

## 🎯 Target

- `log_next_day_volume`: Log-transformed full-day volume for the *next business day*

---

## 🔍 Notes

- The model uses **LightGBM** with `regression` objective and `rmse` metric.
- **TimeSeriesSplit** is used to simulate realistic walk-forward validation.
- Symbol is encoded categorically for a unified cross-symbol model.
