🧠 Simple Approach in KDB (No Integer Programming)
We’ll use linear algebra to compute a least-squares hedge that:

Matches factor exposure as closely as possible

Then optionally keep only the top 5 largest positions

✅ Step-by-Step (Pure q Implementation)
1. Inputs (mock data)
q
Copy
Edit
/ Factor exposure of portfolio (K factors)
b_p: enlist each 0.6 0.8 0.3         / shape: (K x 1)

/ Futures factor exposure matrix (M futures × K factors)
X_fut: flip (1.0 0.3 0.0; 0.2 1.1 0.1; 0.4 0.6 0.8; -0.1 0.2 0.3; 0.5 -0.6 0.1)

/ Factor covariance matrix (K × K)
F: 0.04 0.01 0.00
   0.01 0.03 0.01
   0.00 0.01 0.05
2. Compute Hedge Weights Using Closed-Form Solution
This solves:

min
⁡
ℎ
(
𝑏
𝑝
−
𝑋
𝑓
𝑇
ℎ
)
𝑇
𝐹
(
𝑏
𝑝
−
𝑋
𝑓
𝑇
ℎ
)
h
min
​
 (b 
p
​
 −X 
f
T
​
 h) 
T
 F(b 
p
​
 −X 
f
T
​
 h)
q
Copy
Edit
/ 1. Matrix multiply: Q = X_f F X_f'
Q: X_fut mmu F mmu flip X_fut

/ 2. Vector: c = X_f F b_p
c: X_fut mmu F mmu b_p

/ 3. Solve: h = Q^-1 c
h: inv Q mmu c
✅ h now contains hedge weights for each future

3. Optional: Keep Only Top 5 (Sparse Approximation)
q
Copy
Edit
/ Get top 5 largest abs weights
ix: 5#desc abs each h
topH: h ix
topIndex: ix
Then you can create a sparse hedge vector:

q
Copy
Edit
sparseH: (count h)#0f
sparseH[topIndex]: topH
✅ This gives a hedge vector with only 5 active futures, rest = 0

4. Optional: Scale to Notional
q
Copy
Edit
notional: 10000000f
scale: notional % sum abs sparseH
h_scaled: sparseH * scale
🧾 Summary
Step	What You Do
Q = X_f F X_f'	Build futures covariance
c = X_f F b_p	Build linear projection
h = Q^-1 c	Compute hedge weights
top 5	Sparsify hedge
scale	Convert to dollar weights
