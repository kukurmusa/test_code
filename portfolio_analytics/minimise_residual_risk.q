/ -------------------------------------------
/ Barra single-future hedge evaluator
/ -------------------------------------------

/ Compute optimal hedge ratio β for a future
betaBarra:{[Xp; Xf; F]
  num: Xp mmu F mmu flip enlist Xf;
  den: Xf mmu F mmu flip enlist Xf;
  first num % first den
}

/ Compute residual variance after hedge
residualVarBarra:{[Xp; Xf; F]
  beta: betaBarra[Xp; Xf; F];
  diff: Xp - beta * Xf;
  first diff mmu F mmu flip enlist diff
}

/ Main function: Evaluate hedge candidates
barraSingleHedgeRank:{[Xp; F; Xfuts]
  betas: betaBarra[Xp;]' each Xfuts;
  resVars: residualVarBarra[Xp;]' each Xfuts;
  flip `futureID`beta`residualVar! (til count Xfuts; betas; resVars)
}
