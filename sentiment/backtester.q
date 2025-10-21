//// SENTIMENT DATA TABLE
sentiment:([]
  timestamp:`timestamp$();     / news release time
  sym:`symbol$();              / ticker
  score:`float$();             / -1 to 1
  relevance:`float$();         / Ravenpack relevance score
  novelty:`float$();           / Ravenpack novelty
  category:`symbol$()          / news category
  );

//// TICK DATA (assuming you have this)
trade:([]
  timestamp:`timestamp$();
  sym:`symbol$();
  price:`float$();
  size:`int$();
  exchange:`symbol$()
  );

//// REFERENCE DATA
universe:([]
  sym:`symbol$();
  sector:`symbol$();
  country:`symbol$();
  avgVolume:`float$();         / ADV for position sizing
  tickSize:`float$()
  );

//// EXECUTION TABLE (simulated fills)
executions:([]
  timestamp:`timestamp$();
  sym:`symbol$();
  side:`symbol$();             / `buy or `sell
  qty:`int$();
  price:`float$();
  strategy:`symbol$();         / which strategy generated this
  slippage:`float$()
  );

//// POSITIONS TABLE
positions:([]
  date:`date$();
  sym:`symbol$();
  qty:`int$();
  avgPrice:`float$();
  mktValue:`float$();
  pnl:`float$()
  );
  
  //// Calculate VWAP over a time window
calcVWAP:{[tradeTab; startTime; endTime]
  relevant:select from tradeTab where timestamp within (startTime;endTime);
  vwap:exec wavg[price;size] by sym from relevant;
  vwap
  };

//// Get arrival price (price at decision time)
getArrivalPrice:{[tradeTab; sym; decisionTime]
  / Get last trade before or at decision time
  relevant:select from tradeTab where sym=sym, timestamp<=decisionTime;
  if[0=count relevant; :0n];
  exec last price from relevant
  };

//// Aggregate intraday news sentiment
/ Multiple news items -> aggregate with time decay
aggSentiment:{[sentTab; sym; startTime; endTime; halfLife]
  relevant:select from sentTab where sym=sym, timestamp within (startTime;endTime);
  if[0=count relevant; :0n];
  
  / Time decay weight: more recent news weighted higher
  timeFromEnd:endTime - relevant`timestamp;
  weights:exp[neg timeFromEnd % halfLife];
  
  / Weighted average, also considering relevance
  totalWeight:sum weights * relevant`relevance;
  weightedScore:sum (weights * relevant`relevance * relevant`score);
  
  :weightedScore % totalWeight
  };

//// Check for look-ahead bias
validateTiming:{[newsTime; tradeTime]
  if[tradeTime < newsTime; 
    -1 "WARNING: Look-ahead bias detected! Trade at ",string[tradeTime]," uses news from ",string[newsTime];
  ];
  newsTime < tradeTime
  };
  
  //// Overnight news strategy - optimize entry timing for day
overnightExecStrategy:{[sentTab; tradeTab; targetDate; params]
  
  / Get overnight news (from previous close to open)
  prevClose:targetDate - 1;
  mktOpen:targetDate + 08:00:00.000;  / adjust for EuroStoxx hours
  
  overnightNews:select from sentTab where 
    timestamp within (prevClose+16:30:00.000; mktOpen),
    sym in exec sym from universe;
  
  / Aggregate sentiment by stock
  sentByStock:select avgScore:avg score, 
    maxScore:max score, 
    minScore:min score,
    newsCount:count i,
    avgRelevance:avg relevance 
    by sym from overnightNews;
  
  / Decision: positive sentiment -> delay execution to capture momentum
  /           negative sentiment -> execute early to avoid deterioration
  signals:update executionTime:{[score;mktOpen;mktClose]
    / Positive news: execute later (70% through day)
    / Negative news: execute early (30% through day)  
    / Neutral: VWAP schedule
    timing:0.5 + 0.3 * score;  / maps [-1,1] to [0.2, 0.8]
    :mktOpen + timing * mktClose - mktOpen
    }[avgScore;mktOpen;targetDate+16:30:00.000] 
    from sentByStock;
  
  / Filter for high conviction (|score| > threshold, high relevance)
  signals:select from signals where 
    (abs[avgScore] > params`scoreThreshold),
    (avgRelevance > params`relevanceThreshold),
    (newsCount >= params`minNewsCount);
  
  :signals
  };

//// Simulate execution and calculate slippage vs VWAP
simulateExecution:{[signals; tradeTab; targetDate]
  
  mktOpen:targetDate + 08:00:00.000;
  mktClose:targetDate + 16:30:00.000;
  
  / Calculate VWAP for the day
  dayVWAP:calcVWAP[tradeTab; mktOpen; mktClose];
  
  / Simulate execution at signal time
  execs:update 
    execPrice:getArrivalPrice[tradeTab;;executionTime] each sym,
    vwapPrice:dayVWAP sym,
    slippage:(execPrice - dayVWAP[sym]) % dayVWAP[sym]
    from signals;
  
  / Calculate improvement vs naive VWAP execution
  results:select sym, executionTime, avgScore, 
    execPrice, vwapPrice, slippageBps:10000*slippage,
    pnlBps:neg[10000]*slippage*signum[avgScore]  / positive if we improved
    from execs;
  
  :results
  };
  
  
  
  //// Intraday alpha strategy - take positions based on sentiment
intradayAlphaStrategy:{[sentTab; tradeTab; tradingDate; params]
  
  mktOpen:tradingDate + 08:00:00.000;
  mktClose:tradingDate + 16:30:00.000;
  
  / Define rebalance times (e.g., every 30 minutes)
  rebalTimes:mktOpen + 00:30:00.000 * til floor[(mktClose-mktOpen) % 00:30:00.000];
  
  signals:();
  
  {[sentTab;tradeTab;rebalTime;params]
    
    / Aggregate sentiment from last period (e.g., 1 hour)
    lookback:params`lookbackWindow;
    halfLife:params`sentimentHalfLife;
    
    currentSentiment:{[sentTab;sym;rebalTime;lookback;halfLife]
      aggSentiment[sentTab; sym; rebalTime-lookback; rebalTime; halfLife]
      }[sentTab;;rebalTime;lookback;halfLife] each exec sym from universe;
    
    / Create signal table
    sigs:([] sym:exec sym from universe; sentiment:currentSentiment; rebalTime:rebalTime);
    sigs:delete from sigs where null sentiment;
    
    / Normalize scores and create positions
    sigs:update zScore:(sentiment - avg sentiment) % dev sentiment from sigs;
    
    / Long top decile, short bottom decile
    topThresh:params`topPercentile;
    botThresh:params`botPercentile;
    
    sigs:update signal:{[z;top;bot]
      $[z > top; 1; z < bot; -1; 0]
      }[zScore;topThresh;botThresh] from sigs;
    
    sigs:select from sigs where signal <> 0;
    
    / Position sizing based on signal strength and ADV
    sigs:sigs lj `sym xkey select sym, avgVolume from universe;
    sigs:update 
      targetNotional:params[`totalCapital] * signal * abs[zScore] % sum abs zScore,
      maxNotional:params[`maxAdvPct] * avgVolume  / don't exceed % of ADV
      from sigs;
    
    sigs:update targetNotional:targetNotional & maxNotional from sigs;
    
    :`rebalTime`sym`sentiment`zScore`signal`targetNotional xcols sigs;
    
  }[sentTab;tradeTab;;params] each rebalTimes
  
  :raze signals
  };

//// Backtest the alpha strategy
backtestAlpha:{[signals; tradeTab; startDate; endDate; params]
  
  dates:startDate + til 1 + endDate - startDate;
  dates:dates where (`date$dates) in exec distinct `date$timestamp from tradeTab;
  
  / Initialize portfolio
  portfolio:([sym:exec distinct sym from signals] qty:0; avgPrice:0n; mktValue:0.0);
  
  pnlHistory:();
  
  {[signals;tradeTab;portfolio;tradingDate;params]
    
    / Get signals for this date
    daySignals:select from signals where `date$rebalTime = tradingDate;
    
    / For each rebalance time
    rebalTimes:exec distinct rebalTime from daySignals;
    
    {[signals;tradeTab;portfolio;rebalTime;params]
      
      / Get target positions
      targets:select from signals where rebalTime=rebalTime;
      
      / Current prices
      currentPrices:{[tradeTab;sym;t] getArrivalPrice[tradeTab;sym;t]}[tradeTab;;rebalTime] each exec sym from targets;
      targets:update currentPrice:currentPrices from targets;
      targets:update targetQty:floor targetNotional % currentPrice from targets;
      
      / Calculate required trades
      currentPos:select sym, currentQty:qty from portfolio;
      targets:targets lj `sym xkey currentPos;
      targets:update currentQty:0^currentQty from targets;
      targets:update tradeQty:targetQty - currentQty from targets;
      
      / Simulate fills with slippage
      targets:update 
        fillPrice:currentPrice * 1 + params[`slippageBps]%10000 * signum[tradeQty],
        txnCost:abs[tradeQty] * currentPrice * params[`txnCostBps]%10000
        from targets;
      
      / Update portfolio (would need to implement properly)
      / Store PnL
      
      :targets
      
    }[daySignals;tradeTab;portfolio;;params] each rebalTimes;
    
  }[signals;tradeTab;portfolio;;params] each dates;
  
  :pnlHistory
  };
  
  
  
  //// Calculate strategy statistics
calcStats:{[results]
  stats:`sharpe`maxDD`avgReturn`winRate`totalPnL!(0n;0n;0n;0n;0n);
  
  / Daily returns
  dailyRet:select sum pnlBps by date:`date$timestamp from results;
  
  stats[`avgReturn]:avg dailyRet`pnlBps;
  stats[`sharpe]:avg[dailyRet`pnlBps] % dev dailyRet`pnlBps;
  stats[`totalPnL]:sum dailyRet`pnlBps;
  
  / Max drawdown
  cumRet:sums dailyRet`pnlBps;
  runningMax:cumRet | prior cumRet;
  drawdown:runningMax - cumRet;
  stats[`maxDD]:max drawdown;
  
  / Win rate
  stats[`winRate]:100 * avg dailyRet[`pnlBps] > 0;
  
  :stats
  };

//// Performance attribution
attributePerf:{[results; universe]
  / By sector
  bySector:select 
    totalPnL:sum pnlBps,
    avgPnL:avg pnlBps,
    sharpe:avg[pnlBps] % dev pnlBps,
    tradeCount:count i
    by sector 
    from results lj `sym xkey universe;
  
  / By signal strength
  bySignal:select 
    totalPnL:sum pnlBps,
    avgPnL:avg pnlBps,
    tradeCount:count i
    by signalBucket:10 xbar abs 100*avgScore 
    from results;
  
  `bySector`bySignal!(bySector;bySignal)
  };
  
  
  
  //// Run the backtester
runBacktest:{[startDate; endDate]
  
  / Parameters
  execParams:`scoreThreshold`relevanceThreshold`minNewsCount`slippageBps`txnCostBps!(0.3; 50; 2; 5; 2);
  
  alphaParams:`lookbackWindow`sentimentHalfLife`topPercentile`botPercentile`totalCapital`maxAdvPct`slippageBps`txnCostBps!(01:00:00.000; 00:20:00.000; 1.5; -1.5; 10000000; 0.05; 10; 5);
  
  / Load data (assuming tables are already loaded)
  / sentiment, trade, universe tables
  
  / Strategy 1: Execution timing
  -1 "Running execution timing strategy...";
  dates:startDate + til 1 + endDate - startDate;
  
  execResults:raze {[sentTab;tradeTab;d;params]
    signals:overnightExecStrategy[sentTab; tradeTab; d; params];
    simulateExecution[signals; tradeTab; d]
  }[sentiment;trade;;execParams] each dates;
  
  execStats:calcStats[execResults];
  -1 "Execution Strategy Stats:";
  show execStats;
  
  / Strategy 2: Alpha generation
  -1 "\nRunning alpha strategy...";
  alphaSignals:intradayAlphaStrategy[sentiment; trade; startDate; alphaParams];
  alphaResults:backtestAlpha[alphaSignals; trade; startDate; endDate; alphaParams];
  
  alphaStats:calcStats[alphaResults];
  -1 "Alpha Strategy Stats:";
  show alphaStats;
  
  / Attribution
  attribution:attributePerf[execResults; universe];
  -1 "\nPerformance Attribution:";
  show attribution;
  
  `execResults`execStats`alphaResults`alphaStats`attribution!(execResults;execStats;alphaResults;alphaStats;attribution)
  };

//// Example usage:
/ results:runBacktest[2024.01.01; 2024.12.31]
