
/--------------------------------------------------------------
/ barraPortfolioRisk.q
/ Computes Barra-style portfolio risk, breakdowns, and VaR
/--------------------------------------------------------------

barraPortfolioRisk:{[weights; X; F; specVar; notional]

    / Convert to matrices
    w:enlist weights;
    Xmat:flip X;
    Fmat:flip F;
    D:diag specVar;

    / Compute total covariance matrix: Σ = XFX' + D
    systematicCov:Xmat mmu Fmat mmu flip Xmat;
    totalCov:systematicCov + D;

    / Portfolio variance and volatility
    portVar: w mmu totalCov mmu flip w;
    portVol: sqrt first portVar;

    / Systematic and specific components
    sysVar: w mmu systematicCov mmu flip w;
    specVarTotal: w mmu D mmu flip w;
    sysPct: first sysVar % first portVar;
    specPct: first specVarTotal % first portVar;

    / Portfolio exposure to factors: b = w'X
    b: first w mmu Xmat;

    / Factor contribution to risk: b_i * (Fb)_i
    Fb: Fmat mmu enlist b;
    factorContrib: b * first each Fb;
    factorContribPct: factorContrib % first portVar;

    / MCR = (Σw)_i / σ_p
    sigmaW: totalCov mmu flip w;
    mcr: (first each sigmaW) % portVol;

    / CTR = weight * MCR
    ctr: weights * mcr;

    / Convert all risk numbers to dollar and bps terms
    factorContribUSD: factorContrib * notional;
    factorContribBPS: factorContrib * 10000;

    sysRiskUSD: first sysVar * notional;
    specRiskUSD: first specVarTotal * notional;
    portVolUSD: portVol * notional;

    sysRiskBPS: first sysVar * 10000;
    specRiskBPS: first specVarTotal * 10000;
    portVolBPS: portVol * 10000;

    / MCR and CTR in $
    mcrUSD: mcr * notional;
    ctrUSD: ctr * notional;

    mcrBPS: mcr * 10000;
    ctrBPS: ctr * 10000;

    / Daily volatility and VaR
    dailyVol: portVol % sqrt 252f;
    z95: 1.645;
    z99: 2.326;

    var95: z95 * dailyVol * notional;
    var99: z99 * dailyVol * notional;
    var95bps: var95 % (notional % 10000f);
    var99bps: var99 % (notional % 10000f);

    / Return as dictionary
    (
      `PortfolioVolatility`PortfolioVolatilityUSD`PortfolioVolatilityBPS
        ! (portVol; portVolUSD; portVolBPS);

      `SystematicRisk`SpecificRisk`SystematicRiskUSD`SpecificRiskUSD`SystematicRiskBPS`SpecificRiskBPS
        ! (first sysVar; first specVarTotal; sysRiskUSD; specRiskUSD; sysRiskBPS; specRiskBPS);

      `FactorContribution`FactorContributionUSD`FactorContributionBPS`FactorContributionPct
        ! (factorContrib; factorContribUSD; factorContribBPS; factorContribPct);

      `MCR`MCRUSD`MCRBPS
        ! (mcr; mcrUSD; mcrBPS);

      `CTR`CTRUSD`CTRbps
        ! (ctr; ctrUSD; ctrBPS);

      `VaR_95_USD`VaR_99_USD`VaR_95_BPS`VaR_99_BPS
        ! (var95; var99; var95bps; var99bps)
    )
};



/ Example data
weights: 0.3 0.3 0.2 0.2;
X: (1.2 0.3 0.1; 1.1 0.2 0.2; 0.8 -0.1 0.5; 0.9 0.0 0.7);
F: (0.05 0.01 0.00; 0.01 0.03 0.01; 0.00 0.01 0.04);
specVar: 0.02 0.015 0.025 0.03;
notional: 10000000;

/ Run the function
barraPortfolioRisk[weights; X; F; specVar; notional]
