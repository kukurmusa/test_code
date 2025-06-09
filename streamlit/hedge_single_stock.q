/ --------------------------------------------------
/ Example: Hedge basket using Barra factor exposures
/ --------------------------------------------------

/ 2 positions (A and B)
wA: 500000f
wB: 300000f

/ Factor exposures (3 factors)
XA: 1.1 0.5 -0.2
XB: 0.9 0.3 0.1

/ Hedge basket (C, D, E) exposures
XC: 1.2 0.4 0.0
XD: 0.8 0.6 -0.1
XE: 1.0 0.5 0.2

/ --------------------------------------------------
/ Step 1: Calculate target exposure offset
/ --------------------------------------------------
targetOffset: neg (wA * XA) + (wB * XB)
targetOffset  / should be a 3-element vector

/ --------------------------------------------------
/ Step 2: Create hedge matrix H (3 factors x 3 hedge names)
/ --------------------------------------------------
H: flip (XC; XD; XE)

/ --------------------------------------------------
/ Step 3: Solve for hedge weights
/ Use least squares approach: hedgeWeights = inv(H'H) H' targetOffset
/ --------------------------------------------------
Ht: til 3 each H  / Transpose (3x3)
HtH: Ht mmu H     / (H'H)
invHtH: inv HtH   / Inverse

/ Calculate hedge weights
hedgeWeights: invHtH mmu Ht mmu targetOffset

/ --------------------------------------------------
/ Output
/ --------------------------------------------------
"Target offset:", targetOffset
"Hedge weights for C, D, E:", hedgeWeights

/ Verify final exposure neutralisation
finalExposure: H mmu hedgeWeights
neutralisedExposure: finalExposure + wA*XA + wB*XB
"Final neutralised exposure:", neutralisedExposure


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/ Combined exposure vector
targetOffset: - 910000 440000 -70000

/ Hedge matrix (3x8)
H: flip (
  1.2 0.8 1.0 0.9 1.1 0.85 1.05 0.95;
  0.4 0.7 0.3 0.5 0.6 0.4 0.5 0.45;
  0.0 -0.1 0.2 -0.05 0.1 0.0 -0.1 0.05
)

/ Transpose for least squares
Ht: til 3 each H
HtH: Ht mmu H
invHtH: inv HtH

/ Solve for hedge weights
hedgeWeights: invHtH mmu Ht mmu targetOffset

/ Output
hedgeNames: ("D"; "E"; "F"; "G"; "H"; "I"; "J"; "K")
{show enlist x," hedge notional: ", string hedgeWeights x} each hedgeNames

/ Optional: convert to shares
sharePrices: (10; 12; 15; 8; 20; 11; 13; 9)  / sample prices
hedgeShares: hedgeWeights % sharePrices
"Shares to short for hedge basket:"
{show enlist x," shares: ", string hedgeShares x} each hedgeNames

