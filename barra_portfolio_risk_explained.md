
# Barra Portfolio Risk Calculation in KDB+/q

This document explains the structure and calculations used in the `barraPortfolioRisk` function implemented in KDB+/q. It computes the total risk of a portfolio using a Barra-style factor risk model, and provides detailed breakdowns and representations in dollar and basis point (bps) terms.

---

## Function: `barraPortfolioRisk`

### **Inputs:**

- `weights`: List of portfolio weights (e.g. `[0.3, 0.3, 0.2, 0.2]`)
- `X`: Factor exposure matrix (each row is an asset, columns are factors)
- `F`: Factor covariance matrix (square matrix of factor volatilities/correlations)
- `specVar`: Vector of specific (idiosyncratic) variances per asset
- `notional`: Total value of the portfolio in USD (e.g. `10000000`)

---

## Key Concepts and Formulas

### 1. **Total Covariance Matrix**
\[
\Sigma = X F X^T + D
\]
Where:
- \( X \) is the exposure matrix (assets × factors)
- \( F \) is the factor covariance matrix
- \( D \) is a diagonal matrix of specific variances

---

### 2. **Portfolio Variance and Volatility**
\[
\sigma_p^2 = w^T \Sigma w, \quad \sigma_p = \sqrt{\sigma_p^2}
\]

---

### 3. **Systematic and Specific Risk**
- Systematic:
\[
\text{Systematic Risk} = w^T X F X^T w
\]
- Specific:
\[
\text{Specific Risk} = w^T D w
\]

Their percentages of total risk are:
\[
\text{Systematic %} = \frac{\text{Systematic Risk}}{\text{Total Risk}}, \quad
\text{Specific %} = \frac{\text{Specific Risk}}{\text{Total Risk}}
\]

---

### 4. **Factor Contribution to Risk**
\[
\text{Factor Exposure} = b = w^T X \\
\text{Factor Contribution}_i = b_i \cdot (F b)_i
\]

---

### 5. **Marginal Contribution to Risk (MCR)**
\[
\text{MCR}_i = \frac{(\Sigma w)_i}{\sigma_p}
\]

---

### 6. **Contribution to Risk (CTR)**
\[
\text{CTR}_i = w_i \cdot \text{MCR}_i
\]

---

### 7. **Converting to Dollar and Basis Points**
- Volatility:
\[
\text{Volatility}_\$ = \sigma_p \cdot \text{Notional}, \quad
\text{Volatility}_{bps} = \sigma_p \cdot 10000
\]

- MCR and CTR:
\[
\text{MCR}_\$ = \text{MCR} \cdot \text{Notional}, \quad
\text{MCR}_{bps} = \text{MCR} \cdot 10000
\]

---

### 8. **Value-at-Risk (VaR)**
Assuming normal distribution and daily VaR:
\[
\text{Daily Volatility} = \frac{\sigma_p}{\sqrt{252}}
\]
\[
\text{VaR}_{95\%} = z_{0.95} \cdot \text{Daily Volatility} \cdot \text{Notional}
\]
\[
\text{VaR}_{99\%} = z_{0.99} \cdot \text{Daily Volatility} \cdot \text{Notional}
\]

Where:
- \( z_{0.95} = 1.645 \)
- \( z_{0.99} = 2.326 \)

---

## Output

The function returns a dictionary with the following keys:

- Portfolio volatility, variance in absolute, dollar, and bps terms
- Systematic and specific risk components
- Factor-level contributions
- Marginal and total contributions to risk
- Daily VaR at 95% and 99% in both dollar and bps units

---

## Notes

- All calculations assume annualised volatilities unless specified
- Use consistent units (weights sum to 1, variances in annual terms)
- Factor exposures must be aligned with covariance matrix dimensions

