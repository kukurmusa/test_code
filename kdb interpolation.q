/ Stretch volume curve from 10 to 15 points while keeping sum = 1

/ 1. Original volume curve (must sum to 1)
vc10: 0.05 0.1 0.12 0.08 0.1 0.15 0.1 0.1 0.1 0.1

/ 2. Define x positions for 10 and 15 points (normalised from 0 to 1)
x10: (til 10)%9
x15: (til 15)%14

/ 3. Linear interpolation function
interp:{[xOld;yOld;xNew]
  / For each xNew, find the bracketed (x1,x2) from xOld
  idxs: {first where xOld>=x}' each xNew;
  idxs: idxs - 1;  / ensure we get lower index
  idxs: idxs max 0;  / avoid negative index
  
  x1s: xOld idxs;
  x2s: xOld[idxs+1];
  y1s: yOld idxs;
  y2s: yOld[idxs+1];
  
  fracs: (xNew - x1s) % (x2s - x1s);
  y1s + fracs * (y2s - y1s)
}

/ 4. Interpolate and renormalise
vc15raw: interp[x10; vc10; x15];
vc15: vc15raw % sum vc15raw;

/ 5. Output
(`time`volume)!flip(x15; vc15)