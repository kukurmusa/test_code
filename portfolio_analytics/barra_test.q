/ --- Helper: Diagonal matrix ---
diag:{(count x)#'0N}

barraRiskBreakdownDollar:{
  [weights; X; F; specVar; notional; factorMeta]

  / --- Convert annualised to daily ---
  scale:1 % sqrt 252f;
  Fdaily:F * scale * scale;
  specVard:specVar * scale * scale;

  / --- Shapes ---
  n:count weights;
  w:enlist weights;
  Xmat:flip X;
  D:diag specVard;

  / --- Covariance ---
  sysCov:Xmat mmu Fdaily mmu flip Xmat;
  totalCov:sysCov + D;

  / --- Portfolio Volatility & Risk ---
  portVar:w mmu totalCov mmu flip w;
  portVol:sqrt first portVar;
  portDollarRisk:portVol * notional;

  / --- Asset-level MCR/CTR/Dollar ---
  mcr:(totalCov mmu flip w) % portVol;
  ctr:weights * mcr;
  ctrDollar:ctr * notional;

  / --- Specific/Systematic Risk Components ---
  sysRisk:first (w mmu sysCov mmu flip w);
  specRisk:first (w mmu D mmu flip w);
  sysDollarRisk:sysRisk * notional;
  specDollarRisk:specRisk * notional;

  / --- Factor-Level CTR ($) ---
  factorNames:factorMeta[`factor];
  factorExposure:flip X;
  factorMCR:(factorExposure mmu Fdaily mmu flip factorExposure mmu flip w) % portVol;
  factorCTR:factorMCR * sum each flip factorExposure */: weights;
  factorCTR$:factorCTR * notional;

  / --- Factor Table with Metadata ---
  factorTable:flip `factor`category`CTR`DollarCTR!(
    factorNames;
    factorMeta[`category];
    factorCTR;
    factorCTR$
  );

  / --- Grouped by Factor Category ---
  factorByCat:update DollarCTR:sum DollarCTR by category from factorTable;

  / --- Output Structure ---
  (
    `portfolio`assetBreakdown`factorBreakdown`factorByCategory!(
      (
        `Volatility`Variance`Systematic`Specific`DollarRisk`Systematic$`Specific$!(
          portVol;
          portVar;
          sysRisk;
          specRisk;
          portDollarRisk;
          sysDollarRisk;
          specDollarRisk
        )
      );

      flip `Asset`Weight`MCR`CTR`DollarCTR!(
        til n;
        weights;
        mcr;
        ctr;
        ctrDollar
      );

      factorTable;

      factorByCat
    )
  )
}



weights: 0.2 0.3 0.5
X:(1 0.5 0 0 0; 0.4 1 0.6 0 0; 0.6 0.3 0 1 0.2)
F:5#0.0; F[0 0]:0.04; F[1 1]:0.05; F[2 2]:0.03; F[3 3]:0.02; F[4 4]:0.015
specVar:1.5 2.0 1.0  / Annualised variances (bps^2)
notional:100000000f

factorMeta:([] factor:`Value`Momentum`Tech`UK`France; category:`Style`Style`Industry`Country`Country)
