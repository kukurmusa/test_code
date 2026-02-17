# Multi-Day Market Impact Model Extension

## 1. Motivation

When executing large orders (e.g. 50%+ ADV), single-day execution concentrates participation and drives up temporary market impact due to the concave (square-root) relationship between participation rate and cost. Splitting execution across multiple days exploits this nonlinearity — halving participation doesn't halve impact, but you only need twice as many days to complete. The net effect is lower total cost at the expense of increased timing risk.

This document extends our existing single-period impact model to a multi-day framework, outlines the cost decomposition, derives the optimal daily scheduling, and provides worked numerical examples.

---

## 2. Single-Period Baseline Model

Our current model decomposes execution cost into three components:

$$
C = \underbrace{\frac{s}{2}}_{\text{spread}} + \underbrace{\eta \cdot \sigma \cdot \left(\frac{q}{V}\right)^\alpha}_{\text{temporary impact}} + \underbrace{\gamma \cdot \sigma \cdot \left(\frac{q}{V}\right)^\beta}_{\text{permanent impact}}
$$

Where:

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| $s$ | Bid-ask spread | Stock-specific |
| $\sigma$ | Daily volatility | Stock-specific |
| $q$ | Order quantity (shares) | — |
| $V$ | Average daily volume (ADV) | — |
| $\eta$ | Temporary impact coefficient | Calibrated |
| $\gamma$ | Permanent impact coefficient | Calibrated |
| $\alpha$ | Temporary impact exponent | ~0.5 |
| $\beta$ | Permanent impact exponent | ~0.5 |

The exponents $\alpha \approx \beta \approx 0.5$ reflect the well-documented square-root law of market impact (Almgren et al. 2005, Lillo et al. 2003).

---

## 3. Multi-Day Extension

For an order of total size $Q$ split across $N$ days with daily slices $q_1, q_2, \ldots, q_N$ where $\sum_{i=1}^{N} q_i = Q$:

### 3.1 Spread Cost

Spread is paid on each day's execution independently. Per-share spread cost is unchanged:

$$
C_{\text{spread}} = \sum_{i=1}^{N} \frac{s_i}{2}
$$

This component is **neutral** to the single-day vs. multi-day decision — you pay the same spread per share regardless of how you slice.

### 3.2 Temporary Impact

Temporary impact resets overnight (we assume full reset, $\rho = 0$, as a simplification for liquid names). Each day's temporary cost depends only on that day's participation rate:

$$
C_{\text{temp}} = \sum_{i=1}^{N} \eta \cdot \sigma_i \cdot \left(\frac{q_i}{V_i}\right)^\alpha
$$

This is where the multi-day benefit comes from. Because $\alpha < 1$, the function is concave — splitting participation across days produces a lower total.

**Optional refinement — overnight decay factor $\rho$:** If temporary impact does not fully reset, we can model residual carry-over:

$$
C_{\text{temp},i} = \eta \cdot \sigma_i \cdot \left(\frac{q_i}{V_i}\right)^\alpha + \rho \cdot C_{\text{temp},i-1}^{\text{residual}}
$$

For liquid large-caps, $\rho \approx 0$ is reasonable. For less liquid names, $\rho$ in the range 0.05–0.2 may apply. We recommend starting with $\rho = 0$ and calibrating later.

### 3.3 Permanent Impact

Permanent impact accumulates across days. Day $i$'s execution suffers from: (a) its own permanent impact, and (b) the accumulated permanent shift from all prior days' trading.

The total permanent cost is:

$$
C_{\text{perm}} = \sum_{i=1}^{N} (Q - Q_{i-1}) \cdot \gamma \cdot \sigma_i \cdot \left(\frac{q_i}{V_i}\right)^\beta
$$

Where $Q_{i-1} = \sum_{j=1}^{i-1} q_j$ is the cumulative quantity filled before day $i$, and $(Q - Q_{i-1})$ represents the remaining shares that "pay" for day $i$'s permanent shift.

Early permanent impact is costly because all subsequent slices execute at the shifted price.

### 3.4 Complete Multi-Day Cost Function

Combining all components:

$$
C_{\text{total}} = \sum_{i=1}^{N} \frac{s_i}{2} + \sum_{i=1}^{N} \eta \cdot \sigma_i \cdot \left(\frac{q_i}{V_i}\right)^\alpha + \sum_{i=1}^{N} (Q - Q_{i-1}) \cdot \gamma \cdot \sigma_i \cdot \left(\frac{q_i}{V_i}\right)^\beta
$$

This can be minimised over the daily slices $\{q_i\}$ subject to $\sum q_i = Q$ to find the optimal execution schedule.

---

## 4. Worked Examples

### Assumptions for all examples

| Parameter | Value |
|-----------|-------|
| Stock price | $50 |
| ADV | 1,000,000 shares |
| Daily volatility ($\sigma$) | 2% ($1.00) |
| Spread ($s$) | $0.02 |
| $\eta$ (temp impact coeff) | 0.10 |
| $\gamma$ (perm impact coeff) | 0.05 |
| $\alpha$ (temp exponent) | 0.5 |
| $\beta$ (perm exponent) | 0.5 |

We assume constant $\sigma$, $V$, and $s$ across days for clarity.

### Example 1: Single-Day Execution of 500,000 Shares (50% ADV)

Participation rate: $q/V = 0.50$

| Component | Calculation | Cost ($/share) |
|-----------|-------------|-----------------|
| Spread | $0.02 / 2$ | $0.0100 |
| Temporary | $0.10 \times 1.00 \times \sqrt{0.50}$ | $0.0707 |
| Permanent | $0.05 \times 1.00 \times \sqrt{0.50}$ | $0.0354 |
| **Total** | | **$0.1161** |

**Total dollar cost: 500,000 × $0.1161 = $58,033**

### Example 2: Two-Day Execution — Equal Split (250,000 per day, 25% ADV each)

Participation rate each day: $q/V = 0.25$

**Day 1:**

| Component | Calculation | Cost ($/share) |
|-----------|-------------|-----------------|
| Spread | $0.02 / 2$ | $0.0100 |
| Temporary | $0.10 \times 1.00 \times \sqrt{0.25}$ | $0.0500 |
| Permanent | $0.05 \times 1.00 \times \sqrt{0.25}$ | $0.0250 |

Day 1 cost on 250,000 shares: 250,000 × $0.0850 = $21,250

**Day 2:**

| Component | Calculation | Cost ($/share) |
|-----------|-------------|-----------------|
| Spread | $0.02 / 2$ | $0.0100 |
| Temporary | $0.10 \times 1.00 \times \sqrt{0.25}$ | $0.0500 |
| Permanent (own) | $0.05 \times 1.00 \times \sqrt{0.25}$ | $0.0250 |
| Permanent (carry from Day 1) | $0.05 \times 1.00 \times \sqrt{0.25}$ | $0.0250 |

Day 2 cost on 250,000 shares: 250,000 × $0.1100 = $27,500

Note: Day 2 shares pay for Day 1's permanent impact ($0.025/share) in addition to their own costs.

**Total two-day cost: $21,250 + $27,500 = $48,750**

### Comparison

| Scenario | Total Cost | Cost (bps) | Saving vs 1-Day |
|----------|-----------|------------|-----------------|
| 1-Day (50% ADV) | $58,033 | 23.2 bps | — |
| 2-Day (25% ADV × 2) | $48,750 | 19.5 bps | **$9,283 (16.0%)** |

The saving is driven almost entirely by the temporary impact reduction: $\sqrt{0.25} = 0.50$ vs $\sqrt{0.50} = 0.707$, a 29% drop in per-share temporary impact, applied across all shares.

### Example 3: Three-Day Execution — Equal Split (~167,000 per day, 16.7% ADV)

Participation rate each day: $q/V = 0.167$

| Day | Shares | Temp Impact | Perm Impact (own) | Perm Carry | Spread | Day Cost |
|-----|--------|-------------|-------------------|------------|--------|----------|
| 1 | 166,667 | $0.0408 | $0.0204 | $0.0000 | $0.0100 | $11,872 |
| 2 | 166,667 | $0.0408 | $0.0204 | $0.0204 | $0.0100 | $15,272 |
| 3 | 166,667 | $0.0408 | $0.0204 | $0.0408 | $0.0100 | $18,672 |
| **Total** | | | | | | **$45,816** |

| Scenario | Total Cost | Cost (bps) | Saving vs 1-Day |
|----------|-----------|------------|-----------------|
| 1-Day | $58,033 | 23.2 bps | — |
| 2-Day | $48,750 | 19.5 bps | 16.0% |
| 3-Day | $45,816 | 18.3 bps | **21.1%** |

Diminishing returns are evident — the jump from 1 to 2 days saves ~3.7 bps, while 2 to 3 days saves only ~1.2 bps.

### Example 4: Cost Comparison Across Number of Days

For the same 500,000 share order (50% ADV):

| Days | Participation/Day | Temp ($/sh) | Total Cost | bps | Marginal Saving |
|------|-------------------|-------------|-----------|-----|-----------------|
| 1 | 50.0% | $0.0707 | $58,033 | 23.2 | — |
| 2 | 25.0% | $0.0500 | $48,750 | 19.5 | 3.7 bps |
| 3 | 16.7% | $0.0408 | $45,816 | 18.3 | 1.2 bps |
| 5 | 10.0% | $0.0316 | $43,301 | 17.3 | 0.5 bps/day |
| 10 | 5.0% | $0.0224 | $41,833 | 16.7 | 0.1 bps/day |

The marginal benefit of additional days declines rapidly, while **timing risk increases linearly** with the number of days. This creates a natural optimal horizon.

---

## 5. Timing Risk — The Missing Piece

The cost model above ignores **timing risk**: the risk that the price moves adversely while we're still executing. Over $N$ days, the variance of execution cost due to price drift is approximately:

$$
\text{Timing Risk (std dev)} \approx \sigma \cdot \sqrt{N} \cdot \frac{Q_{\text{remaining}}}{Q}
$$

For our 500,000 share example with $\sigma$ = 2%:

| Days | Impact Cost (bps) | Timing Risk 1σ (bps) | Risk-Adjusted Cost (bps) |
|------|-------------------|----------------------|--------------------------|
| 1 | 23.2 | 0 | 23.2 |
| 2 | 19.5 | ~142 | 161 |
| 3 | 18.3 | ~173 | 191 |
| 5 | 17.3 | ~224 | 241 |

Timing risk dominates quickly. The optimal number of days balances the declining marginal impact savings against the increasing timing risk. This tradeoff is controlled by a **risk aversion parameter** $\lambda$ in the Almgren-Chriss framework:

$$
\min_{\{q_i\}} \left[ C_{\text{total}}(\{q_i\}) + \lambda \cdot \text{Var}(\{q_i\}) \right]
$$

For most institutional risk aversion levels, the optimal horizon for a 50% ADV order is typically **2–3 days**.

---

## 6. Optimal Daily Scheduling

For equal volatility and volume across days, the key tradeoff in scheduling is:

- **Front-loading** reduces timing risk (less remaining exposure to adverse moves) but concentrates impact
- **Back-loading** reduces per-day impact but carries more permanent impact from earlier days and faces more uncertainty

The Almgren-Chriss solution yields a schedule that is **roughly uniform but slightly front-loaded**. In practice, for a 2-day execution of 50% ADV:

| Strategy | Day 1 | Day 2 | Total Cost | Timing Risk |
|----------|-------|-------|-----------|-------------|
| Equal split | 250K (25%) | 250K (25%) | $48,750 | Moderate |
| Front-loaded (60/40) | 300K (30%) | 200K (20%) | $49,891 | Lower |
| Back-loaded (40/60) | 200K (20%) | 300K (30%) | $49,891 | Higher |

The equal split is close to optimal for cost alone; slight front-loading is preferred once risk aversion is included.

---

## 7. Calibration Requirements

Extending from the single-day model, the additional parameter to calibrate is:

| Parameter | Description | How to Calibrate |
|-----------|-------------|------------------|
| $\rho$ | Overnight decay of temporary impact | Regress Day 2 open price vs Day 1 close, conditioned on Day 1 participation rate |

All other parameters ($\eta$, $\gamma$, $\alpha$, $\beta$) carry over from the existing single-day calibration.

**Calibration approach for $\rho$:**
1. Identify multi-day parent orders from execution data
2. Measure the opening price on Day $N+1$ relative to Day $N$'s close
3. Condition on Day $N$'s participation rate and subtract expected overnight drift
4. The residual attributable to prior-day impact gives an estimate of $\rho$

---

## 8. Key Takeaways

1. **Multi-day execution is cheaper** for large orders due to the concave (square-root) impact function. A 50% ADV order saves ~16% in impact costs by splitting over 2 days.

2. **Diminishing returns** set in quickly — the 1-to-2 day improvement is much larger than 2-to-3 days. For most practical purposes, 2–3 days is the sweet spot for orders in the 30–60% ADV range.

3. **Timing risk is the binding constraint**, not impact cost. The optimal horizon is set by the tradeoff between declining marginal impact savings and linearly increasing price uncertainty.

4. **Optimal scheduling is slightly front-loaded** once risk aversion is included, but close to a uniform split for moderate risk aversion.

5. **The framework requires only one new parameter** ($\rho$, overnight decay) beyond the existing single-day calibration.

---

## 9. Academic References

- **Almgren, R. & Chriss, N. (2000)** — "Optimal Execution of Portfolio Transactions", *Journal of Risk*, 3(2), pp. 5–39. Foundational paper establishing the permanent + temporary impact framework and multi-period optimal scheduling with risk aversion. See **Section 2** (Efficient Frontier of Optimal Execution), specifically **Section 2.2** (Explicit construction of optimal strategies) and **Section 2.3** (The half-life of a trade).

- **Almgren, R. (2003)** — "Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk", *Applied Mathematical Finance*, 10(1), pp. 1–18. Extends the framework to power-law (concave) impact functions, which is what makes multi-day splitting beneficial. Introduces the size-dependent "characteristic time" for optimal trading.

- **Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005)** — "Direct Estimation of Equity Market Impact", *Risk*. Empirical calibration of the $\sigma \cdot (q/V)^\alpha$ functional form, establishing $\alpha \approx 0.5$ (the square-root law) from real execution data.

---

## 10. Next Steps

1. **Calibrate $\rho$** using our existing multi-day parent order data
2. **Build the optimizer** to solve for optimal $\{q_i\}$ given order size, risk aversion, and stock-specific parameters
3. **Backtest** the multi-day scheduling against historical execution data to validate cost savings
4. **Integrate timing risk** via the Almgren-Chriss mean-variance framework to produce risk-adjusted optimal horizons
5. **Extend to portfolio context** — when executing basket orders, cross-asset correlation affects the timing risk component
