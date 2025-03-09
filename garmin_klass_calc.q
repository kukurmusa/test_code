// Load 1-minute OHLC data (Assuming table 'ohlc' with columns: time, open, high, low, close)
\l stock_data.q

// Define the Garman-Klass variance formula
gkVar: { 
    (0.5 * (x[`high] - x[`low])^2) - ((2 * log 2 - 1) * (x[`close] - x[`open])^2) 
}

// Apply Garman-Klass volatility per 1-minute interval
ohlc:update GK_var: gkVar each ohlc from ohlc
ohlc:update GK_vol_1min: sqrt GK_var from ohlc

// Convert 1-minute volatility to annualised volatility
T: 252 * 6.5 * 60;  // Total 1-minute periods in a year
ohlc:update GK_vol_annual: GK_vol_1min * sqrt T from ohlc

// Show the results
select time, open, high, low, close, GK_vol_1min, GK_vol_annual from ohlc
