diag:{(count x)#'0N}; diag[x] +\: x

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

    / Portfolio exposure to factors: b = w'X
    b: first w mmu Xmat;
    Fb: Fmat mmu enlist b;
    factorContrib: b * first each Fb;
    factorContribPct: factorContrib % first portVar;

    / MCR = (Σw)_i / σ_p, CTR = weight * MCR
    sigmaW: totalCov mmu flip w;
    mcr: (first each sigmaW) % portVol;
    ctr: weights * mcr;

    / Convert to dollar and bps
    factorContribUSD: factorContrib * notional;
    factorContribBPS: factorContrib * 10000;

    mcrUSD: mcr * notional;
    ctrUSD: ctr * notional;
    mcrBPS: mcr * 10000;
    ctrBPS: ctr * 10000;

    sysRiskUSD: first sysVar * notional;
    specRiskUSD: first specVarTotal * notional;
    portVolUSD: portVol * notional;
    sysRiskBPS: first sysVar * 10000;
    specRiskBPS: first specVarTotal * 10000;
    portVolBPS: portVol * 10000;

    / Daily volatility and VaR
    dailyVol: portVol % sqrt 252f;
    z95: 1.645; z99: 2.326;
    var95: z95 * dailyVol * notional;
    var99: z99 * dailyVol * notional;
    var95bps: var95 % (notional % 10000f);
    var99bps: var99 % (notional % 10000f);

    / Create tables
    factorTable: ([] 
        FactorID: til count factorContrib;
        Exposure: b;
        FactorContrib: factorContrib;
        FactorContribUSD: factorContribUSD;
        FactorContribBPS: factorContribBPS;
        FactorContribPct: factorContribPct
    );

    assetTable: ([] 
        AssetID: til count weights;
        Weight: weights;
        MCR: mcr;
        MCRUSD: mcrUSD;
        MCRBPS: mcrBPS;
        CTR: ctr;
        CTRUSD: ctrUSD;
        CTRBPS: ctrBPS
    );

    / Return
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
        ! (var95; var99; var95bps; var99bps);

      `FactorTable`AssetTable ! (factorTable; assetTable)
    )
};
