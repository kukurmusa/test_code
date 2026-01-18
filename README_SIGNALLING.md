# Formal Signalling Score (SIG) – kdb+/q Specification

## Overview
This document describes a **formal signalling score (SIG)** designed to detect when an execution algorithm
(VWAP / POV) is **moving the market due to its presence**, rather than genuine market drift.

The score decomposes signalling into:
1. **Active-phase drift** (price moves against you while active)
2. **Post-cancel / stop reversion** (market reverts when you leave)
3. **Footprint intensity** (how detectable your trading behaviour is)

All metrics are **direction-aware**, **volatility-normalised**, and suitable for production monitoring.

---

## Core Idea

For a BUY order:
- Price drifting **up while active** is adverse
- Price **falling after cancel/completion** indicates signalling

For a SELL order, the logic is symmetric.

SIG is designed to explain why:
- VWAP can look acceptable
- Arrival Cost (especially notional-weighted) deteriorates badly

---

## Input Tables (Minimum)

### parentOrder
| column | type | description |
|------|------|-------------|
| orderId | symbol | parent order id |
| symbol | symbol | instrument |
| side | symbol | `BUY` or `SELL` |
| startTime | timestamp | algo start |
| endTime | timestamp | algo end |
| cancelTime | timestamp | null if completed |
| qty | float | total quantity |
| notional | float | executed notional |
| algo | symbol | VWAP / POV |
| participationTarget | float | target POV |

### childEvent
| column | type |
|-------|------|
| orderId | symbol |
| time | timestamp |
| eventType | symbol | `NEW` `REPLACE` `CANCEL` `FILL` |
| price | float |
| qty | float |
| venue | symbol |
| displayed | boolean |

### mkt (snapshots, 1s–5s)
| column | type |
|------|------|
| time | timestamp |
| symbol | symbol |
| bid | float |
| ask | float |
| bidSize | float |
| askSize | float |

Derived:
- mid = (bid + ask) / 2
- spread = ask - bid

---

## Component Scores

### 1. Active Drift (Z_active)
Measures adverse signed price drift **while the order is active**, volatility-normalised.

High values imply:
> “The market moved against me because I was trading.”

---

### 2. Cancel / Stop Reversion (Z_reversion)
Event-study style metric around cancel time.

Strong signalling signature:
- adverse drift **before cancel**
- mean reversion **after cancel**

This is the strongest causal indicator of signalling.

---

### 3. Footprint Intensity (Z_footprint)
Behavioural detectability:
- Time at BBO
- Replace rate
- Child size entropy

Low entropy + high visibility = easy to detect.

---

## Final Score

SIG = 0.4 × Z_active  
    + 0.4 × Z_reversion  
    + 0.2 × Z_footprint  

Interpretation:
- SIG < 1.0 → normal
- 1.0–2.0 → watch
- 2.0–3.0 → strong signalling
- > 3.0 → urgent

---

## Usage

Recommended workflow:
1. Compute SIG daily in kdb
2. Join with Arrival Cost diagnostics
3. Drill down when:
   - SIG high & AC bad → footprint problem
   - SIG low & AC bad → regime / market issue

---

## Next Extensions
- Completion-based reversion score
- Venue-level signalling attribution
- Real-time SIG alerts
- Use SIG as penalty term in execution optimiser

