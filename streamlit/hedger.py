import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gurobipy as gp
from gurobipy import GRB


# ===============================
# Gurobi hedge optimiser (generic)
# ===============================
def optimise_hedge_gurobi(
    B_b,
    B_h,
    w_b,
    Cov_F,
    adv,
    portfolio_notional,
    max_hedges=3,
    adv_frac=0.20,          # e.g. 20% of ADV
    max_leverage=5.0,       # sum |w_h| <= max_leverage
    time_limit=60,
    mip_gap=0.005,
    verbose=True,
):
    """
    Solve the systematic (or factor-subset) risk-minimising hedge problem with Gurobi (MIQP).

    B_b : (n_assets, k_factors)
    B_h : (m_futures, k_factors)
    w_b : (n_assets,)
    Cov_F : (k_factors, k_factors)
    adv : (m_futures,)
    """

    n_assets, k_factors = B_b.shape
    m_futures, k_factors_h = B_h.shape
    assert k_factors == k_factors_h, "B_b and B_h must have same number of factors"

    # --- Clean / regularise covariance ---
    Cov_F = 0.5 * (Cov_F + Cov_F.T)
    eps = 1e-4
    Cov_F = Cov_F + np.eye(k_factors) * eps

    # --- Precompute exposures & quadratic terms ---
    # net_exposure = exposure_b + B_h.T @ w_h
    # risk = || Cov_F @ net_exposure ||^2 = ||A w_h + b||^2
    exposure_b = B_b.T @ w_b                      # (k,)
    A = Cov_F @ B_h.T                             # (k, m)
    b = Cov_F @ exposure_b                        # (k,)

    Q = 2.0 * (A.T @ A)                           # (m, m) PSD
    c = 2.0 * (A.T @ b)                           # (m,)

    # ADV-based max weight per future
    adv_limit = adv_frac * adv / portfolio_notional   # (m,)
    adv_limit = np.asarray(adv_limit, dtype=float)

    # --- Create model ---
    model = gp.Model("HedgeOptimiser")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit

    # --- Variables ---
    w_h = model.addVars(m_futures, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w_h")
    z   = model.addVars(m_futures, vtype=GRB.BINARY, name="z")
    w_abs = model.addVars(m_futures, lb=0.0, name="w_abs")   # |w_h|

    # --- Constraints ---

    # link weights to selection & ADV limits: -adv_lim*z <= w_h <= adv_lim*z
    for i in range(m_futures):
        model.addConstr(w_h[i] <=  adv_limit[i] * z[i], name=f"adv_ub_{i}")
        model.addConstr(w_h[i] >= -adv_limit[i] * z[i], name=f"adv_lb_{i}")

    # absolute value linearisation for leverage
    for i in range(m_futures):
        model.addConstr(w_abs[i] >=  w_h[i], name=f"abs_pos_{i}")
        model.addConstr(w_abs[i] >= -w_h[i], name=f"abs_neg_{i}")

    # max number of hedge instruments
    model.addConstr(gp.quicksum(z[i] for i in range(m_futures)) <= max_hedges,
                    name="max_hedges")

    # leverage constraint: sum |w_h| <= max_leverage
    model.addConstr(gp.quicksum(w_abs[i] for i in range(m_futures)) <= max_leverage,
                    name="max_leverage")

    # --- Objective: minimise ||A w_h + b||^2 = 0.5 w^T Q w + c^T w + const ---
    quad_expr = gp.QuadExpr()

    # 0.5 * w^T Q w
    for i in range(m_futures):
        for j in range(m_futures):
            if Q[i, j] != 0.0:
                quad_expr += 0.5 * Q[i, j] * w_h[i] * w_h[j]

    # linear term c^T w
    for i in range(m_futures):
        if c[i] != 0.0:
            quad_expr += c[i] * w_h[i]

    model.setObjective(quad_expr, GRB.MINIMIZE)

    # --- Solve ---
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi did not find an optimal solution. Status = {model.Status}")

    w_opt = np.array([w_h[i].X for i in range(m_futures)])
    z_opt = np.array([z[i].X for i in range(m_futures)])

    df_result = pd.DataFrame({
        "Future": [f"H{i+1}" for i in range(m_futures)],
        "Weight": w_opt,
        "Selected": (z_opt > 0.5).astype(int),
        "ADV": adv,
        "ADV_limit_weight": adv_limit,
        "Notional": w_opt * portfolio_notional,
        "%ADV_used": np.where(adv > 0, np.abs(w_opt * portfolio_notional) / adv, 0.0),
    })

    return w_opt, z_opt, df_result, model.ObjVal


# ===============================
# Risk analytics / plotting
# ===============================
def run_risk_analytics(edited_df, B_b_target, B_h_target, w_b, Cov_F_target,
                       selected_factor_names, sim_T=100):
    include_mask = edited_df["Include"].values
    if include_mask.sum() == 0:
        st.warning("No hedge instruments included – nothing to analyse.")
        return

    w_h_sel = edited_df["Weight"].values[include_mask]
    B_h_sel = B_h_target[include_mask, :]  # (m_sel, k)

    # basket & hedge exposures
    exposure_b = B_b_target.T @ w_b               # (k,)
    exposure_h = B_h_sel.T @ w_h_sel              # (k,)
    exposure_after = exposure_b + exposure_h

    # risk before / after
    risk_before = np.linalg.norm(Cov_F_target @ exposure_b) ** 2
    risk_after = np.linalg.norm(Cov_F_target @ exposure_after) ** 2

    st.subheader("📊 Factor Risk Before vs After Hedge")
    before_vec = (Cov_F_target @ exposure_b)
    after_vec = (Cov_F_target @ exposure_after)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=selected_factor_names,
        y=before_vec**2,
        name="Before"
    ))
    fig.add_trace(go.Bar(
        x=selected_factor_names,
        y=after_vec**2,
        name="After"
    ))
    fig.update_layout(barmode="group", yaxis_title="Variance Contribution")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total Target Risk Before", f"{risk_before:.4f}")
    col2.metric("Total Target Risk After", f"{risk_after:.4f}")

    # -------- Risk contribution per hedge instrument --------
    st.subheader("🔍 Risk Contribution per Hedge Instrument")
    contrib_rows = []
    for i, fut_row in enumerate(edited_df[include_mask].itertuples(index=False)):
        expo_i = B_h_sel[i, :] * fut_row.Weight   # factor exposure from that hedge
        rc_i = np.linalg.norm(Cov_F_target @ expo_i) ** 2
        contrib_rows.append({
            "Future": fut_row.Future,
            "Weight": fut_row.Weight,
            "RiskContribution": rc_i
        })
    contrib_df = pd.DataFrame(contrib_rows)
    st.dataframe(contrib_df, use_container_width=True)

    # -------- Simulated performance (factor-driven dummy returns) --------
    st.subheader("📈 Simulated Hedge Performance (Factor-driven dummy returns)")
    k = len(selected_factor_names)
    T = sim_T

    # simulate factor returns
    factor_returns = np.random.normal(0, 0.01, size=(T, k))

    # daily returns
    r_basket = factor_returns @ exposure_b
    r_hedge = factor_returns @ exposure_h
    r_net = r_basket + r_hedge

    cum_b = np.cumsum(r_basket)
    cum_h = np.cumsum(r_hedge)
    cum_n = np.cumsum(r_net)

    perf_df = pd.DataFrame({
        "t": np.arange(T),
        "Basket": cum_b,
        "Hedge": cum_h,
        "Net": cum_n,
    })

    fig_p = go.Figure()
    for col in ["Basket", "Hedge", "Net"]:
        fig_p.add_trace(go.Scatter(
            x=perf_df["t"],
            y=perf_df[col],
            mode="lines",
            name=col
        ))
    fig_p.update_layout(
        title="Simulated Cumulative Return (Dummy)",
        xaxis_title="Time",
        yaxis_title="Cumulative Return"
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # -------- Summary stats --------
    st.subheader("📌 Performance Summary Stats")
    stats = {}
    for name, series in zip(
        ["Basket", "Hedge", "Net"],
        [r_basket, r_hedge, r_net]
    ):
        vol = np.std(series)
        sharpe = np.mean(series) / vol if vol > 0 else np.nan
        cum = np.cumsum(series)
        dd = (np.maximum.accumulate(cum) - cum).max()
        stats[name] = {"Volatility": vol, "Sharpe": sharpe, "MaxDrawdown": dd}
    stats_df = pd.DataFrame(stats).T
    st.dataframe(stats_df.style.format("{:.4f}"))

    # -------- Factor attribution for net PnL --------
    st.subheader("📎 Factor Attribution of Net Return")
    factor_pnl = factor_returns * exposure_after  # (T, k)
    avg_contrib = factor_pnl.mean(axis=0)
    attr_df = pd.Series(avg_contrib, index=selected_factor_names)
    st.bar_chart(attr_df)


# ===============================
# Streamlit App
# ===============================

st.set_page_config("Systematic / Factor / Beta Hedge Optimiser (Gurobi)", layout="wide")
st.title("🛡️ Hedge Optimiser (Barra Factor vs ETF Beta) – Gurobi")

# ---- Sidebar: model choice ----
st.sidebar.header("🧠 Hedge Model")
model_type = st.sidebar.radio(
    "Hedge model",
    ["Barra factor hedge", "ETF beta hedge"],
)

# ensure session resets when switching model types
if "model_type" not in st.session_state:
    st.session_state["model_type"] = model_type
elif st.session_state["model_type"] != model_type:
    # clear only model-dependent keys
    for key in ["hedge_df", "B_b_target", "B_h_target", "w_b",
                "Cov_F_target", "selected_factor_names", "portfolio_notional",
                "sim_T"]:
        st.session_state.pop(key, None)
    st.session_state["model_type"] = model_type

st.sidebar.header("📥 Basket & Notional")
portfolio_notional = st.sidebar.number_input(
    "Portfolio Notional (£)",
    min_value=1_000_000,
    max_value=500_000_000,
    value=10_000_000,
    step=1_000_000,
)

st.sidebar.header("⚙️ Hedge Constraints")
max_hedges = st.sidebar.slider("Max Hedge Instruments", 1, 20, 3)
max_adv_pct = st.sidebar.slider("Max % of ADV per Hedge", 5, 100, 20) / 100
max_leverage = st.sidebar.slider("Max Sum |Weight| of Hedges", 1.0, 20.0, 5.0, step=0.5)
sim_T = st.sidebar.slider("Simulation Length (days)", 50, 500, 200, step=50)

# -------------------------
# MODE 1: Barra factor hedge
# -------------------------
if model_type == "Barra factor hedge":
    st.sidebar.header("Barra Inputs (dummy)")
    n_assets = st.sidebar.slider("Number of Basket Assets", 50, 600, 200, step=10)
    m_futures = st.sidebar.slider("Number of Hedge Futures", 5, 40, 15, step=1)
    k_factors = st.sidebar.slider("Number of Factors", 10, 100, 25, step=1)

    factor_names = [f"Factor_{i}" for i in range(k_factors)]
    hedge_symbols = [f"H{i+1}" for i in range(m_futures)]

    hedge_type = st.sidebar.radio("Hedge Type", ["Systematic (All Factors)", "Factor-Specific"])
    if hedge_type == "Factor-Specific":
        selected_factor_names = st.sidebar.multiselect(
            "Select Factors to Hedge",
            factor_names,
            default=factor_names[:3],
        )
    else:
        selected_factor_names = factor_names

    target_indices = [factor_names.index(f) for f in selected_factor_names]

    # ----- Dummy Barra-like inputs -----
    np.random.seed(42)
    w_b = np.random.rand(n_assets)
    w_b /= w_b.sum()

    B_b = np.random.randn(n_assets, k_factors)
    B_h = np.random.randn(m_futures, k_factors)

    A_cov = np.random.randn(k_factors, k_factors)
    Cov_F = A_cov.T @ A_cov
    Cov_F = 0.5 * (Cov_F + Cov_F.T) + np.eye(k_factors) * 1e-4

    adv = np.random.uniform(1e6, 10e6, size=m_futures)

    # restrict to selected factors
    B_b_target = B_b[:, target_indices]          # (n, k_sel)
    B_h_target = B_h[:, target_indices]          # (m, k_sel)
    Cov_F_target = Cov_F[np.ix_(target_indices, target_indices)]

    st.markdown("### 1️⃣ Optimise Hedge (Barra factor model, Gurobi)")

    if st.button("🔍 Run Hedge Optimisation"):
        try:
            w_opt, z_opt, df_hedge, obj_val = optimise_hedge_gurobi(
                B_b=B_b_target,
                B_h=B_h_target,
                w_b=w_b,
                Cov_F=Cov_F_target,
                adv=adv,
                portfolio_notional=portfolio_notional,
                max_hedges=max_hedges,
                adv_frac=max_adv_pct,
                max_leverage=max_leverage,
                time_limit=60,
                mip_gap=0.005,
                verbose=True,
            )

            df_hedge["Include"] = df_hedge["Selected"] == 1

            st.session_state["hedge_df"] = df_hedge
            st.session_state["B_b_target"] = B_b_target
            st.session_state["B_h_target"] = B_h_target
            st.session_state["w_b"] = w_b
            st.session_state["Cov_F_target"] = Cov_F_target
            st.session_state["selected_factor_names"] = selected_factor_names
            st.session_state["portfolio_notional"] = portfolio_notional
            st.session_state["sim_T"] = sim_T

            st.success(f"Hedge optimisation complete. Objective value: {obj_val:.6f}")

        except Exception as e:
            st.error(f"Gurobi optimisation failed: {e}")

# -------------------------
# MODE 2: ETF beta hedge
# -------------------------
else:
    st.sidebar.header("ETF Beta Inputs (dummy)")
    m_futures = st.sidebar.slider("Number of ETF Hedges", 2, 20, 5, step=1)

    # initialise / resize ETF beta table
    if "beta_etf_df" not in st.session_state or \
       st.session_state["beta_etf_df"].shape[0] != m_futures:
        np.random.seed(123)
        df_beta = pd.DataFrame({
            "Future": [f"ETF_{i+1}" for i in range(m_futures)],
            "Beta": np.random.uniform(0.5, 1.5, size=m_futures),
            "ADV": np.random.uniform(1e6, 10e6, size=m_futures),
        })
        st.session_state["beta_etf_df"] = df_beta

    st.markdown("### 1️⃣ Define ETF Beta Universe")
    beta_df = st.data_editor(
        st.session_state["beta_etf_df"],
        use_container_width=True,
        num_rows="fixed",
        key="beta_etf_editor",
    )
    st.session_state["beta_etf_df"] = beta_df

    # single-factor: 'Beta' factor
    selected_factor_names = ["Beta"]
    k_factors = 1

    # Simplify basket: 1 asset with beta 1.0 and weight 1.0 (you can replace with real basket beta)
    B_b_target = np.array([[1.0]])        # (1,1)
    w_b = np.array([1.0])                 # basket as single exposure
    B_h_target = beta_df["Beta"].values.reshape(-1, 1)  # (m,1)
    Cov_F_target = np.array([[1.0]])      # variance scale for Beta factor (drops out in relative terms)
    adv = beta_df["ADV"].values

    if st.button("🔍 Run ETF Beta Hedge Optimisation"):
        try:
            w_opt, z_opt, df_hedge, obj_val = optimise_hedge_gurobi(
                B_b=B_b_target,
                B_h=B_h_target,
                w_b=w_b,
                Cov_F=Cov_F_target,
                adv=adv,
                portfolio_notional=portfolio_notional,
                max_hedges=max_hedges,
                adv_frac=max_adv_pct,
                max_leverage=max_leverage,
                time_limit=60,
                mip_gap=0.005,
                verbose=True,
            )

            # Rename Future column to match ETF names
            df_hedge["Future"] = beta_df["Future"].values
            df_hedge["Include"] = df_hedge["Selected"] == 1

            st.session_state["hedge_df"] = df_hedge
            st.session_state["B_b_target"] = B_b_target
            st.session_state["B_h_target"] = B_h_target
            st.session_state["w_b"] = w_b
            st.session_state["Cov_F_target"] = Cov_F_target
            st.session_state["selected_factor_names"] = selected_factor_names
            st.session_state["portfolio_notional"] = portfolio_notional
            st.session_state["sim_T"] = sim_T

            st.success(f"ETF beta hedge optimisation complete. Objective value: {obj_val:.6f}")

        except Exception as e:
            st.error(f"Gurobi optimisation failed: {e}")

# -------------------------
# Common: table + analytics
# -------------------------
if "hedge_df" in st.session_state:
    st.markdown("### 2️⃣ Review / Edit Hedge Recommendation")

    edited_df = st.data_editor(
        st.session_state["hedge_df"],
        use_container_width=True,
        num_rows="fixed",
        key="hedge_editor",
    )

    st.markdown("### 3️⃣ Post-Hedge Risk Analytics & Simulation")

    if st.button("📉 Run Risk Analytics on Current Selection"):
        run_risk_analytics(
            edited_df=edited_df,
            B_b_target=st.session_state["B_b_target"],
            B_h_target=st.session_state["B_h_target"],
            w_b=st.session_state["w_b"],
            Cov_F_target=st.session_state["Cov_F_target"],
            selected_factor_names=st.session_state["selected_factor_names"],
            sim_T=st.session_state["sim_T"],
        )

        # download selected hedge
        csv = edited_df[edited_df["Include"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Selected Hedge as CSV",
            data=csv,
            file_name="selected_hedge_gurobi.csv",
            mime="text/csv",
        )
else:
    st.info("Run the optimisation first to see hedge suggestions.")
