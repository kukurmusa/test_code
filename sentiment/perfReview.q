/ VWAP Algo Performance Analysis in KDB/Q
/ Slippage decomposition and diagnostic analysis

/ ============================================================================
/ 1. GENERATE DUMMY DATA
/ ============================================================================

/ Order-level data
orders: ([] 
    orderID: 1000 + til 200;
    date: 200#2024.10.01 + til 90;
    symbol: 200?`AAPL`MSFT`GOOGL`AMZN`TSLA`META`NVDA`JPM`BAC`WFC;
    side: 200?`BUY`SELL;
    shares: 200?10000 20000 50000 100000 200000;
    arrivalTime: 200#09:00:00.000;
    arrivalPrice: 100 + 200?50.0;
    execVWAP: 100 + 200?50.0;
    benchmarkVWAP: 100 + 200?50.0;
    orderDuration: 200?360 390 420;  / minutes (6-7 hours)
    pctADV: 200?5.0 10.0 15.0 20.0 25.0;
    completionRate: 95.0 + 200?5.0
    )

/ Add some correlation to make realistic
orders: update execVWAP: arrivalPrice + (0.5*pctADV*0.01*arrivalPrice) + (-0.02 + 200?0.04)*arrivalPrice from orders
orders: update benchmarkVWAP: arrivalPrice + (-0.015 + 200?0.03)*arrivalPrice from orders

/ Fill-level data (10 fills per order on average)
nFills: 2000
fills: ([] 
    fillID: til nFills;
    orderID: nFills?exec orderID from orders;
    fillTime: 09:00:00.000 + nFills?06:30:00.000;
    fillPrice: 100 + nFills?50.0;
    fillShares: nFills?500 1000 2000 5000;
    bid: 100 + nFills?50.0;
    ask: 100 + nFills?50.0;
    isPassive: nFills?0b 1b;
    venue: nFills?`NASDAQ`NYSE`BATS`IEX`EDGX
    )

/ Ensure bid < ask and realistic spreads
fills: update ask: bid + 0.01 + (nFills?0.05) from fills
fills: update fillPrice: bid + (ask-bid)*?[isPassive;0.3 + nFills?0.3;0.7 + nFills?0.3] from fills

/ Minute-by-minute market data (simplified)
minutes: raze {[oid] 
    o: first select from orders where orderID=oid;
    times: o[`arrivalTime] + 00:01:00.000 * til 390;
    drift: sums (-0.001 + (count times)?0.002) * o[`arrivalPrice];
    ([] orderID: oid; 
        minute: times; 
        midPrice: o[`arrivalPrice] + drift;
        volume: 1000 + (count times)?5000)
    } each exec orderID from orders

/ ============================================================================
/ 2. CORE CALCULATIONS - SLIPPAGE DECOMPOSITION
/ ============================================================================

/ Calculate basic slippage metrics
orders: update 
    arrivalSlippage: 10000*(execVWAP - arrivalPrice)%arrivalPrice,
    vwapSlippage: 10000*(execVWAP - benchmarkVWAP)%benchmarkVWAP,
    marketDrift: 10000*(benchmarkVWAP - arrivalPrice)%arrivalPrice
    from orders

/ Implementation shortfall (your performance vs benchmark)
orders: update implementationShortfall: arrivalSlippage - marketDrift from orders

/ Calculate first fill time and delay cost
firstFills: select firstFillTime: min fillTime, 
                   firstFillPrice: first fillPrice 
            by orderID from fills

orders: orders lj firstFills
orders: update delayCost: 10000*(firstFillPrice - arrivalPrice)%arrivalPrice from orders

/ Calculate spread costs at fill level
fills: update midAtFill: (bid + ask)%2 from fills
fills: update spreadCostPerFill: fillPrice - midAtFill from fills

/ Join order side to fills to calculate signed spread cost
fills: fills lj `orderID xkey select orderID, side from orders
fills: update spreadCostPerFill: ?[side=`SELL; neg spreadCostPerFill; spreadCostPerFill] from fills

/ Aggregate spread cost to order level
spreadCosts: select totalSpreadCost: sum spreadCostPerFill * fillShares,
                    totalFillShares: sum fillShares
             by orderID from fills

orders: orders lj spreadCosts
orders: update spreadCostBps: 10000*totalSpreadCost%totalFillShares%arrivalPrice from orders
orders: update spreadCostBps: 0^spreadCostBps from orders  / handle nulls

/ Calculate passive fill rate
passiveStats: select passiveShares: sum ?[isPassive; fillShares; 0],
                     totalFillShares: sum fillShares
              by orderID from fills

orders: orders lj passiveStats  
orders: update passiveFillRate: 100*passiveShares%totalFillShares from orders
orders: update passiveFillRate: 0^passiveFillRate from orders

/ Calculate adverse selection on passive fills
passiveFills: select from fills where isPassive

/ Get price 1 minute after each passive fill (simplified - using next minute)
passiveFills: passiveFills lj `orderID`minute xcols 
    update minute: fillTime from passiveFills

/ For this example, create synthetic post-fill prices
passiveFills: update priceAfter1min: fillPrice + (-0.02 + (count passiveFills)?0.04) from passiveFills

/ Calculate adverse selection (negative is bad)
passiveFills: passiveFills lj `orderID xkey select orderID, side from orders
passiveFills: update adverseSelectionPerFill: 
    ?[side=`BUY; priceAfter1min - fillPrice; fillPrice - priceAfter1min] 
    from passiveFills

adverseSelection: select avgAdverseSelection: wavg[adverseSelectionPerFill; fillShares],
                         passiveShares: sum fillShares
                  by orderID from passiveFills

orders: orders lj adverseSelection
orders: update adverseSelectionBps: 10000*avgAdverseSelection%arrivalPrice from orders
orders: update adverseSelectionBps: 0^adverseSelectionBps from orders

/ Calculate timing cost (residual)
orders: update timingCost: implementationShortfall - spreadCostBps - delayCost from orders

/ Calculate participation rate (approximate)
orders: update participationRate: (shares%orderDuration)%100 from orders  / shares per minute relative to baseline

/ ============================================================================
/ 3. PERFORMANCE SEGMENTATION ANALYSIS
/ ============================================================================

/ Summary statistics
summaryStats: select 
    count: count orderID,
    avgArrivalSlip: avg arrivalSlippage,
    avgVWAPSlip: avg vwapSlippage,
    avgImplShortfall: avg implementationShortfall,
    avgMarketDrift: avg marketDrift,
    avgSpreadCost: avg spreadCostBps,
    avgDelayCost: avg delayCost,
    avgTimingCost: avg timingCost,
    avgPassiveRate: avg passiveFillRate,
    avgAdverseSelection: avg adverseSelectionBps
    from orders

/ Quartile analysis by implementation shortfall
orders: update implShortfallQuartile: ?[implementationShortfall < -2; `Q1_Best; 
                                       ?[implementationShortfall < 0; `Q2_Good;
                                       ?[implementationShortfall < 2; `Q3_Poor; `Q4_Worst]]] 
        from orders

quartileAnalysis: select 
    count: count orderID,
    avgImplShortfall: avg implementationShortfall,
    avgSpreadCost: avg spreadCostBps,
    avgPassiveRate: avg passiveFillRate,
    avgAdverseSelection: avg adverseSelectionBps,
    avgPctADV: avg pctADV,
    avgTimingCost: avg timingCost
    by implShortfallQuartile from orders

/ Performance by order size buckets
orders: update sizeCategory: ?[pctADV < 10; `Small;
                              ?[pctADV < 20; `Medium; `Large]]
        from orders

sizeAnalysis: select 
    count: count orderID,
    avgPctADV: avg pctADV,
    avgImplShortfall: avg implementationShortfall,
    avgSpreadCost: avg spreadCostBps,
    avgPassiveRate: avg passiveFillRate,
    avgAdverseSelection: avg adverseSelectionBps
    by sizeCategory from orders

/ Performance by symbol
symbolAnalysis: select 
    count: count orderID,
    avgImplShortfall: avg implementationShortfall,
    avgArrivalSlip: avg arrivalSlippage,
    avgSpreadCost: avg spreadCostBps,
    avgPassiveRate: avg passiveFillRate
    by symbol from orders

/ ============================================================================
/ 4. DIAGNOSTIC FUNCTIONS
/ ============================================================================

/ Function: Identify worst performing orders
worstOrders: {[n]
    n sublist `implementationShortfall xasc 
        select orderID, symbol, date, shares, pctADV, 
               implementationShortfall, spreadCostBps, passiveFillRate, 
               adverseSelectionBps, timingCost 
        from orders
    }

/ Function: Identify best performing orders  
bestOrders: {[n]
    n sublist `implementationShortfall xdesc 
        select orderID, symbol, date, shares, pctADV,
               implementationShortfall, spreadCostBps, passiveFillRate,
               adverseSelectionBps, timingCost
        from orders
    }

/ Function: Compare top vs bottom quartile
compareQuartiles: {[]
    top: first select from quartileAnalysis where implShortfallQuartile=`Q1_Best;
    bottom: first select from quartileAnalysis where implShortfallQuartile=`Q4_Worst;
    
    ([] metric: `implShortfall`spreadCost`passiveRate`adverseSelection`pctADV`timingCost;
        topQuartile: (top`avgImplShortfall; top`avgSpreadCost; top`avgPassiveRate; 
                     top`avgAdverseSelection; top`avgPctADV; top`avgTimingCost);
        bottomQuartile: (bottom`avgImplShortfall; bottom`avgSpreadCost; bottom`avgPassiveRate;
                        bottom`avgAdverseSelection; bottom`avgPctADV; bottom`avgTimingCost);
        difference: (bottom[`avgImplShortfall] - top`avgImplShortfall;
                    bottom[`avgSpreadCost] - top`avgSpreadCost;
                    bottom[`avgPassiveRate] - top`avgPassiveRate;
                    bottom[`avgAdverseSelection] - top`avgAdverseSelection;
                    bottom[`avgPctADV] - top`avgPctADV;
                    bottom[`avgTimingCost] - top`avgTimingCost))
    }

/ Function: Calculate correlation matrix for key metrics
correlationAnalysis: {[]
    metrics: flip `implShortfall`spreadCost`passiveRate`adverseSelection`pctADV`timingCost!
             flip orders[`implementationShortfall`spreadCostBps`passiveFillRate`adverseSelectionBps`pctADV`timingCost];
    
    / Simple correlation calculation
    corrMatrix: {[x;y] (avg[x*y] - (avg x)*avg y) % sqrt[(avg[x*x] - (avg x)*avg x) * (avg[y*y] - (avg y)*avg y)]}
    
    / Build correlation table
    names: cols metrics;
    corrs: names!{[metrics;n1] names!{[metrics;n1;n2] corrMatrix[metrics[n1]; metrics[n2]]}[metrics;n1;] each names}[metrics;] each names;
    corrs
    }

/ ============================================================================
/ 5. REPORTING FUNCTIONS
/ ============================================================================

/ Executive summary report
executiveSummary: {[]
    -1 "\n========================================";
    -1 "VWAP ALGO PERFORMANCE - EXECUTIVE SUMMARY";
    -1 "========================================\n";
    
    -1 "Overall Performance Metrics:";
    -1 "  Avg Implementation Shortfall: ", string[summaryStats`avgImplShortfall], " bps";
    -1 "  Avg Arrival Slippage: ", string[summaryStats`avgArrivalSlip], " bps";
    -1 "  Avg VWAP Slippage: ", string[summaryStats`avgVWAPSlip], " bps";
    -1 "  Avg Market Drift: ", string[summaryStats`avgMarketDrift], " bps\n";
    
    -1 "Cost Breakdown:";
    -1 "  Spread Cost: ", string[summaryStats`avgSpreadCost], " bps";
    -1 "  Delay Cost: ", string[summaryStats`avgDelayCost], " bps";
    -1 "  Timing Cost: ", string[summaryStats`avgTimingCost], " bps";
    -1 "  Adverse Selection: ", string[summaryStats`avgAdverseSelection], " bps\n";
    
    -1 "Execution Characteristics:";
    -1 "  Passive Fill Rate: ", string[summaryStats`avgPassiveRate], " %";
    -1 "  Total Orders Analyzed: ", string[summaryStats`count], "\n";
    
    -1 "========================================\n";
    }

/ Print all analysis
printAnalysis: {[]
    executiveSummary[];
    -1 "Quartile Analysis:";
    show quartileAnalysis;
    -1 "\nSize Category Analysis:";
    show sizeAnalysis;
    -1 "\nSymbol Analysis:";
    show symbolAnalysis;
    -1 "\nTop vs Bottom Quartile Comparison:";
    show compareQuartiles[];
    -1 "\nWorst 10 Orders:";
    show worstOrders[10];
    }

/ ============================================================================
/ 6. RUN ANALYSIS
/ ============================================================================

/ Execute the analysis
printAnalysis[]

/ Additional queries you can run:
/ show summaryStats
/ show worstOrders[20]
/ show bestOrders[20]
/ show select from orders where implementationShortfall > 5
/ show select avg implementationShortfall by sizeCategory, symbol from orders
