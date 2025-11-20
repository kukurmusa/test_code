/ ============================================================================
/ VWAP FOOTPRINT ANALYZER - KDB+/Q IMPLEMENTATION
/ ============================================================================

/ Load embedPy for Python integration (visualization)
\l p.q

/ ============================================================================
/ UTILITY FUNCTIONS
/ ============================================================================

/ Generate time buckets (1-minute intervals from market open to close)
genTimeBuckets:{[d] 
    start:d+09:30:00.000;
    end:d+16:00:00.000;
    start+00:01:00.000*til 390
    };

/ U-shaped intraday volume curve (high at open/close, low midday)
uShapedCurve:{[n]
    x:((til n)%n);
    curve:2.5*((x*x)-(x)+0.6);
    curve+n?0.1;
    curve:curve|0.3;
    curve%avg curve
    };

/ ============================================================================
/ MOCK DATA GENERATOR
/ ============================================================================

/ Symbol parameters
symbolParams:([] 
    symbol:`AAPL`MSFT`GOOGL`TSLA`JPM;
    price:180 380 140 240 155f;
    volatility:0.02 0.018 0.022 0.035 0.015;
    spreadBps:1.0 1.2 1.5 2.0 0.8;
    avgVolume:1000000 800000 600000 1200000 500000
    );

/ Generate one day of tick data for a symbol
genDayData:{[sym;dt;isTrade;tradeIntensity;tradeStartPct;tradeEndPct]
    params:select from symbolParams where symbol=sym;
    p:first params`price;
    vol:first params`volatility;
    spd:first params`spreadBps;
    avgVol:first params`avgVolume;
    
    / Generate volume curve
    nMinutes:390;
    baseCurve:uShapedCurve[nMinutes];
    
    / Add footprint if trade day
    volMultiplier:nMinutes#1f;
    if[isTrade;
        startIdx:floor nMinutes*tradeStartPct;
        endIdx:floor nMinutes*tradeEndPct;
        volMultiplier[startIdx+til endIdx-startIdx]:tradeIntensity;
        / Add leakage
        if[startIdx>10; volMultiplier[startIdx-10+til 10]:1.1];
        if[endIdx<nMinutes-10; volMultiplier[endIdx+til 10]:1.1];
    ];
    baseCurve:baseCurve*volMultiplier;
    
    / Generate ticks
    ticks:raze {[sym;dt;p;vol;spd;avgVol;baseCurve;minute]
        timestamp:dt+09:30:00.000+00:01:00.000*minute;
        nTicks:1|floor 5*baseCurve[minute]*1+0.5-1?1f;
        
        / Generate tick data
        ([] 
            symbol:nTicks#sym;
            date:nTicks#dt;
            timestamp:timestamp+nTicks?00:01:00.000;
            price:p+cumsum nTicks?vol*p*0.01;
            volume:"j"$avgVol*baseCurve[minute]%nMinutes%5*exp nTicks?0.8;
            bid:0f;
            ask:0f;
            bid_size:0j;
            ask_size:0j;
            trade_direction:0j;
            quote_update:nTicks?2
        )
    }[sym;dt;p;vol;spd;avgVol;baseCurve] each til nMinutes;
    
    / Calculate bid/ask from price
    ticks:update 
        spread:price*spd*0.0001*(1+1.5-baseCurve[`int$(`time$timestamp-dt)%60000]),
        bid:price-spread%2,
        ask:price+spread%2
    from ticks;
    
    / Trade direction (biased on trade days)
    ticks:update 
        trade_direction:?[isTrade and 
            ((`time$timestamp-dt)>=09:30:00.000+`time$nMinutes*tradeStartPct*60000) and
            ((`time$timestamp-dt)<09:30:00.000+`time$nMinutes*tradeEndPct*60000);
            {x?(-1;1)}'[55 45];
            {x?(-1;1)}'[50 50]]
    from ticks;
    
    / Depth (inverse to volume curve)
    ticks:update 
        bid_size:"j"$exp 8+1?1f,
        ask_size:"j"$exp 8+1?1f
    from ticks;
    
    / Increase bid depth on trade days
    ticks:update bid_size:"j"$bid_size*1.15 
        from ticks 
        where isTrade, 
            (`time$timestamp-date)>=09:30:00.000+`time$nMinutes*tradeStartPct*60000,
            (`time$timestamp-date)<09:30:00.000+`time$nMinutes*tradeEndPct*60000;
    
    ticks
    };

/ Generate multi-day, multi-symbol data
genMultiDayData:{[symbols;startDate;nDays;tradeDaysPerSymbol]
    / Generate business days
    dates:startDate+til nDays;
    dates:dates where 5>(`week$dates) mod 7;
    
    / Generate data for each symbol and date
    data:raze {[symbols;dates;tradeDaysPerSymbol;sym]
        tradeDays:tradeDaysPerSymbol[sym];
        raze {[sym;tradeDays;dt]
            isTrade:dt in tradeDays;
            intensity:$[isTrade; 1.3+0.5?1f; 1f];
            startPct:$[isTrade; 0.2+0.2?1f; 0.3];
            endPct:$[isTrade; 0.6+0.2?1f; 0.7];
            genDayData[sym;dt;isTrade;intensity;startPct;endPct]
        }[sym;tradeDays] each dates
    }[symbols;dates;tradeDaysPerSymbol] each symbols;
    
    data
    };

/ ============================================================================
/ METRIC CALCULATION
/ ============================================================================

/ Calculate all metrics for time buckets
calcBucketMetrics:{[data]
    / Add time bucket (round to minute)
    data:update timeBucket:01:00t xbar `time$timestamp-date from data;
    
    / Calculate metrics by bucket
    metrics:select 
        volume:sum volume,
        trade_count:count i,
        vwap:volume wavg price,
        volatility:dev price,
        spread:avg ask-bid,
        bid_depth:avg bid_size,
        ask_depth:avg ask_size,
        buy_volume:sum volume where trade_direction=1,
        sell_volume:sum volume where trade_direction=-1,
        quote_intensity:sum quote_update,
        first_price:first price,
        last_price:last price
    by date, timeBucket from data;
    
    / Calculate derived metrics
    metrics:update 
        spread_pct:100*spread%vwap,
        depth_imbalance:(bid_depth-ask_depth)%(bid_depth+ask_depth),
        trade_imbalance:(buy_volume-sell_volume)%(buy_volume+sell_volume),
        price_impact:abs[last_price-first_price]%first_price%(volume%1000000)
    from metrics;
    
    / Calculate price returns (vs previous bucket)
    metrics:update price_return:(vwap-prev vwap)%prev vwap from metrics;
    
    / Calculate effective spread (simplified)
    metrics:update effective_spread:2*spread from metrics;
    
    metrics
    };

/ ============================================================================
/ BASELINE CREATION
/ ============================================================================

/ Create baseline from non-trading days
createBaseline:{[data;tradeDates;currentDate;symbol;lookbackDays]
    / Filter for this symbol
    symData:select from data where symbol=symbol;
    
    / Get prior dates excluding trade days
    allDates:asc distinct symData`date;
    allDates:allDates where allDates<currentDate;
    nonTradeDates:allDates where not allDates in tradeDates;
    baselineDates:neg[lookbackDays]#nonTradeDates;
    
    if[lookbackDays>count baselineDates;
        -1"Warning: Only ",string[count baselineDates]," non-trading days available for ",string[symbol];
    ];
    
    / Get baseline data
    baselineData:select from symData where date in baselineDates;
    
    / Calculate metrics for baseline
    baselineMetrics:calcBucketMetrics[baselineData];
    
    / Aggregate baseline (mean, std, median by time bucket)
    baselineAgg:select 
        volume_mean:avg volume,
        volume_std:dev volume,
        volume_median:med volume,
        trade_count_mean:avg trade_count,
        trade_count_std:dev trade_count,
        vwap_mean:avg vwap,
        vwap_std:dev vwap,
        price_return_mean:avg price_return,
        price_return_std:dev price_return,
        volatility_mean:avg volatility,
        volatility_std:dev volatility,
        spread_mean:avg spread,
        spread_std:dev spread,
        depth_imbalance_mean:avg depth_imbalance,
        depth_imbalance_std:dev depth_imbalance,
        trade_imbalance_mean:avg trade_imbalance,
        trade_imbalance_std:dev trade_imbalance,
        price_impact_mean:avg price_impact,
        price_impact_std:dev price_impact
    by timeBucket from baselineMetrics;
    
    (baselineAgg;baselineMetrics)
    };

/ ============================================================================
/ SIMILARITY MEASURES
/ ============================================================================

/ Pearson correlation
pearsonCorr:{[x;y]
    n:count x;
    if[n<2; :0n];
    xBar:avg x;
    yBar:avg y;
    num:sum (x-xBar)*y-yBar;
    denom:sqrt[sum (x-xBar)*x-xBar]*sqrt sum (y-yBar)*y-yBar;
    $[0=denom; 0n; num%denom]
    };

/ Spearman rank correlation
spearmanCorr:{[x;y]
    if[2>count x; :0n];
    rx:rank x;
    ry:rank y;
    pearsonCorr[rx;ry]
    };

/ Euclidean distance (normalized)
euclideanDist:{[x;y]
    xNorm:x%sum x;
    yNorm:y%sum y;
    sqrt sum (xNorm-yNorm)*xNorm-yNorm
    };

/ Cosine similarity
cosineSim:{[x;y]
    num:sum x*y;
    denom:sqrt[sum x*x]*sqrt sum y*y;
    $[0=denom; 0n; num%denom]
    };

/ KL Divergence
klDivergence:{[x;y]
    eps:1e-10;
    xNorm:(x%sum x)+eps;
    yNorm:(y%sum y)+eps;
    sum xNorm*log xNorm%yNorm
    };

/ Z-scores
calcZScores:{[tradeVals;baselineMean;baselineStd]
    (tradeVals-baselineMean)%baselineStd+1e-10
    };

/ Calculate all similarity metrics for one metric
calcSimilarity:{[tradeVals;baselineMean;baselineStd]
    / Normalize for some measures
    xNorm:tradeVals%sum tradeVals+1e-10;
    yNorm:baselineMean%sum baselineMean+1e-10;
    
    / Calculate z-scores
    zScores:calcZScores[tradeVals;baselineMean;baselineStd];
    
    (!) . flip (
        (`pearson; pearsonCorr[tradeVals;baselineMean]);
        (`spearman; spearmanCorr[tradeVals;baselineMean]);
        (`euclidean; euclideanDist[tradeVals;baselineMean]);
        (`cosine_sim; 1-cosineSim[tradeVals;baselineMean]);
        (`kl_divergence; klDivergence[xNorm;yNorm]);
        (`mean_z_score; avg abs zScores);
        (`max_z_score; max abs zScores)
    )
    };

/ ============================================================================
/ FOOTPRINT ANALYSIS
/ ============================================================================

/ Metric weights for composite score
metricWeights:(`volume`trade_count`price_return`volatility`spread`depth_imbalance`trade_imbalance`price_impact)!(0.20 0.10 0.15 0.15 0.10 0.10 0.10 0.10);

/ Analyze full footprint
analyzeFootprint:{[tradeDayProfile;baselineAgg]
    / Merge trade day and baseline
    merged:tradeDayProfile lj 1!baselineAgg;
    
    / Define metrics to analyze
    metrics:`volume`trade_count`price_return`volatility`spread`depth_imbalance`trade_imbalance`price_impact;
    
    / Calculate similarity for each metric
    results:(!) . flip {[merged;metric]
        tradeCol:metric;
        meanCol:`$string[metric],"_mean";
        stdCol:`$string[metric],"_std";
        
        if[all (tradeCol;meanCol;stdCol) in cols merged;
            similarity:calcSimilarity[merged[tradeCol];merged[meanCol];merged[stdCol]];
            (metric; similarity)
        ]
    }[merged] each metrics;
    
    (results;merged)
    };

/ Compute composite detectability score
computeCompositeScore:{[metricResults]
    / Calculate weighted correlation
    weightedCorr:sum {[metricResults;metric;weight]
        if[metric in key metricResults;
            corr:metricResults[metric][`pearson];
            if[not null corr; :weight*corr];
        ];
        0f
    }[metricResults;;] .'flip (key;value)@\:metricWeights;
    
    totalWeight:sum value metricWeights;
    weightedCorr:weightedCorr%totalWeight;
    
    / Find suspicious metrics
    suspicious:raze {[metricResults;metric]
        if[metric in key metricResults;
            result:metricResults[metric];
            corr:result`pearson;
            if[(not null corr) and (corr<0.7);
                :enlist `metric`correlation`mean_z_score!(metric;corr;result`mean_z_score)
            ];
        ];
        ()
    }[metricResults] each key metricWeights;
    
    / Detectability score (0-100)
    detectability:100*1-weightedCorr;
    
    `weighted_correlation`detectability_score`suspicious_metrics!(weightedCorr;detectability;suspicious)
    };

/ ============================================================================
/ REPORTING
/ ============================================================================

/ Generate text report
generateReport:{[symbol;tradeDate;metricResults;compositeScore]
    -1"";
    -1 80#"=";
    -1"VWAP FOOTPRINT ANALYSIS - ",string[symbol]," - ",string tradeDate;
    -1 80#"=";
    -1"";
    
    detectability:compositeScore`detectability_score;
    -1"Composite Detectability Score: ",string[detectability]," / 100";
    -1"";
    
    status:$[detectability<20; "EXCELLENT: Footprint blends in very well";
            detectability<40; "GOOD: Footprint reasonably disguised";
            detectability<60; "MODERATE: Some detectability - consider randomization";
            "HIGH RISK: Footprint is highly detectable - likely exploitable!"];
    -1 status;
    -1"";
    
    -1 80#"-";
    -1"METRIC-BY-METRIC ANALYSIS";
    -1 80#"-";
    -1"";
    
    / Sort metrics by correlation (worst first)
    metrics:key metricResults;
    correlations:{x`pearson} each metricResults metrics;
    sorted:metrics iasc correlations;
    
    {[metricResults;metric]
        result:metricResults metric;
        corr:result`pearson;
        zScore:result`mean_z_score;
        eucl:result`euclidean;
        
        status:$[corr>0.8;"✅";corr>0.6;"⚠️";"🚨"];
        
        -1 status," ",upper string metric;
        -1"   Correlation:    ",string corr;
        -1"   Mean Z-score:   ",string zScore;
        -1"   Euclidean dist: ",string eucl;
        -1"";
    }[metricResults] each sorted;
    
    suspicious:compositeScore`suspicious_metrics;
    if[count suspicious;
        -1 80#"-";
        -1"SUSPICIOUS METRICS (requiring attention)";
        -1 80#"-";
        {-1"  • ",string[x`metric],": correlation=",string[x`correlation],", z-score=",string[x`mean_z_score]} each suspicious;
        -1"";
    ];
    
    -1 80#"=";
    -1"";
    };

/ ============================================================================
/ PYTHON VISUALIZATION (via embedPy)
/ ============================================================================

/ Initialize Python
.py.set[`np;.p.import`numpy];
.py.set[`pd;.p.import`pandas];
.py.set[`plt;.p.import`matplotlib.pyplot];
.py.set[`sns;.p.import`seaborn];

/ Plot comprehensive footprint
plotFootprint:{[merged;metricResults;symbol;tradeDate]
    / Convert to pandas DataFrame
    df:.p.import[`pandas;`:DataFrame;merged];
    
    / Define metrics to plot
    metrics:`volume`volatility`spread`depth_imbalance`trade_imbalance`price_impact;
    
    / Create plot
    .py.set[`df;df];
    .py.set[`metrics;metrics];
    .py.set[`symbol;symbol];
    .py.set[`tradeDate;string tradeDate];
    .py.set[`metricResults;metricResults];
    
    .p.e"
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 4*len(metrics)))

for idx, metric in enumerate(metrics):
    # Left: Time series
    ax1 = axes[idx, 0]
    
    trade_col = metric
    baseline_col = metric + '_mean'
    std_col = metric + '_std'
    
    x = range(len(df))
    
    if baseline_col in df.columns:
        baseline = df[baseline_col].values
        ax1.plot(x, baseline, label='Baseline (21-day avg)', linewidth=2, alpha=0.7, color='blue')
        
        if std_col in df.columns:
            std = df[std_col].values
            ax1.fill_between(x, baseline-std, baseline+std, alpha=0.2, color='blue')
    
    if trade_col in df.columns:
        trade = df[trade_col].values
        ax1.plot(x, trade, label='Trade Day', linewidth=2, alpha=0.8, color='red')
    
    # Add correlation
    if metric in metricResults:
        corr = metricResults[metric]['pearson']
        ax1.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_title(f'{metric.replace(\"_\", \" \").title()} - Time Series')
    ax1.set_xlabel('Time Bucket (minutes)')
    ax1.set_ylabel(metric)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Distribution
    ax2 = axes[idx, 1]
    
    if baseline_col in df.columns and trade_col in df.columns:
        ax2.hist(df[baseline_col].dropna(), bins=20, alpha=0.5, 
                label='Baseline', color='blue', density=True)
        ax2.hist(df[trade_col].dropna(), bins=20, alpha=0.5,
                label='Trade Day', color='red', density=True)
    
    ax2.set_title(f'{metric.replace(\"_\", \" \").title()} - Distribution')
    ax2.set_xlabel(metric)
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.suptitle(f'Footprint Analysis - {symbol} - {tradeDate}', fontsize=16, y=1.001)
plt.tight_layout()
plt.savefig(f'footprint_{symbol}_{tradeDate}.png', dpi=100, bbox_inches='tight')
plt.close()
";
    
    -1"Chart saved: footprint_",string[symbol],"_",string[tradeDate],".png";
    };

/ Plot multi-symbol heatmap
plotHeatmap:{[allResults]
    / Convert results to table
    resultsTable:flip `symbol`date`detectability!(
        first each flip key allResults;
        last each flip key allResults;
        {x`detectability_score} each value allResults
    );
    
    symbols:asc distinct resultsTable`symbol;
    dates:asc distinct resultsTable`date;
    
    / Create matrix
    matrix:(count[symbols];count[dates])#0n;
    {[resultsTable;matrix;symbols;dates;i]
        sym:symbols i;
        symData:select from resultsTable where symbol=sym;
        {[matrix;dates;i;j;d]
            matrix[i;j]:d`detectability;
        }[matrix;dates;i;;] .' flip (dates?symData`date;symData`detectability)
    }[resultsTable;matrix;symbols;dates] each til count symbols;
    
    / Convert to pandas
    df:.p.import[`pandas;`:DataFrame][matrix];
    .py.set[`df;df];
    .py.set[`symbols;string symbols];
    .py.set[`dates;string dates];
    
    .p.e"
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(figsize=(max(12, len(dates)*0.8), max(6, len(symbols)*0.6)))

# Create mask for NaN
mask = np.isnan(df.values)

sns.heatmap(df.values,
            xticklabels=dates,
            yticklabels=symbols,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            center=50,
            vmin=0, vmax=100,
            mask=mask,
            ax=ax,
            cbar_kws={'label': 'Detectability Score (0-100)'})

ax.set_title('Multi-Symbol Footprint Detectability Heatmap\\nGreen=Well Disguised, Red=Highly Detectable')
ax.set_xlabel('Trade Date')
ax.set_ylabel('Symbol')

plt.tight_layout()
plt.savefig('footprint_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
";
    
    -1"Heatmap saved: footprint_heatmap.png";
    };

/ ============================================================================
/ MAIN WORKFLOW
/ ============================================================================

/ Run complete analysis
runAnalysis:{[data;symbol;tradeDate;tradeDates;lookbackDays]
    / Create baseline
    (baselineAgg;baselineMetrics):createBaseline[data;tradeDates;tradeDate;symbol;lookbackDays];
    
    if[null baselineAgg; :(::)];
    
    / Get trade day data
    tradeDayData:select from data where symbol=symbol, date=tradeDate;
    
    if[0=count tradeDayData; 
        -1"No data for ",string[symbol]," on ",string tradeDate;
        :(::)
    ];
    
    / Calculate trade day metrics
    tradeDayMetrics:calcBucketMetrics[tradeDayData];
    
    / Analyze footprint
    (metricResults;merged):analyzeFootprint[tradeDayMetrics;baselineAgg];
    
    / Compute composite score
    compositeScore:computeCompositeScore[metricResults];
    
    / Generate report
    generateReport[symbol;tradeDate;metricResults;compositeScore];
    
    / Plot (if embedPy available)
    if[`p in key`;
        plotFootprint[merged;metricResults;symbol;tradeDate];
    ];
    
    (metricResults;compositeScore;merged)
    };

/ ============================================================================
/ EXAMPLE USAGE
/ ============================================================================

runExample:{[]
    -1"Generating mock data...";
    
    / Parameters
    symbols:`AAPL`MSFT`GOOGL`TSLA;
    startDate:2024.10.01;
    nDays:35;
    
    / Define trade days per symbol
    tradeDaysPerSymbol:`AAPL`MSFT`GOOGL`TSLA!(
        2024.10.15 2024.10.18 2024.10.22 2024.10.25 2024.10.29;
        2024.10.16 2024.10.21 2024.10.28 2024.10.30;
        2024.10.14 2024.10.17 2024.10.21 2024.10.24 2024.10.28 2024.10.31;
        2024.10.18 2024.10.23 2024.10.30
    );
    
    / Generate data
    data:genMultiDayData[symbols;startDate;nDays;tradeDaysPerSymbol];
    
    -1"Generated ",string[count data]," ticks";
    -1"Date range: ",string[min data`date]," to ",string[max data`date];
    -1"";
    
    / Run analysis for each symbol and trade date
    allResults:();
    
    {[data;tradeDaysPerSymbol;sym]
        tradeDates:tradeDaysPerSymbol sym;
        -1"";
        -1 80#"=";
        -1"ANALYZING ",string[sym]," - ",string[count tradeDates]," trade days";
        -1 80#"=";
        
        {[data;tradeDates;sym;dt]
            result:runAnalysis[data;sym;dt;tradeDates;15];
            if[not null result;
                allResults,:enlist[(sym;dt)]!enlist result[1];
            ];
        }[data;tradeDates;sym] each tradeDates;
    }[data;tradeDaysPerSymbol] each symbols;
    
    / Generate summary heatmap
    if[(`p in key`) and 0<count allResults;
        -1"";
        -1"Generating summary heatmap...";
        plotHeatmap[allResults];
    ];
    
    / Summary statistics
    -1"";
    -1 80#"=";
    -1"SUMMARY STATISTICS";
    -1 80#"=";
    -1"";
    
    {[allResults;sym]
        symScores:{x`detectability_score} each allResults where (first each key allResults)=sym;
        if[count symScores;
            -1"";
            -1 string sym,":";
            -1"  Mean detectability:  ",string avg symScores;
            -1"  Min detectability:   ",string min symScores;
            -1"  Max detectability:   ",string max symScores;
            -1"  Std deviation:       ",string dev symScores;
            highRisk:sum symScores>60;
            if[highRisk>0;
                -1"  High risk days:      ",string[highRisk],"/",string count symScores;
            ];
        ];
    }[allResults] each symbols;
    
    -1"";
    -1 80#"=";
    -1"ANALYSIS COMPLETE!";
    -1 80#"=";
    -1"";
    -1"Analyzed ",string[count allResults]," symbol-date combinations";
    
    allResults
    };

/ ============================================================================
/ PRODUCTION INTEGRATION EXAMPLES
/ ============================================================================

/ Save baseline to disk for reuse
saveBaseline:{[baseline;symbol;date]
    `:baselines/ set (`$string[symbol],"_",string date;baseline);
    };

/ Load baseline from disk
loadBaseline:{[symbol;date]
    get `:baselines/`$string[symbol],"_",string date
    };

/ Real-time monitoring during execution
monitorRealTime:{[liveData;baseline;symbol]
    / Calculate current metrics
    currentMetrics:calcBucketMetrics[liveData];
    
    / Compare to baseline
    (results;merged):analyzeFootprint[currentMetrics;baseline];
    
    / Check for alerts
    compositeScore:computeCompositeScore[results];
    detectability:compositeScore`detectability_score;
    
    if[detectability>60;
        -1"ALERT: High detectability detected (",string[detectability],") for ",string symbol;
    ];
    
    compositeScore
    };

/ Batch analysis for multiple days
batchAnalysis:{[data;symbol;tradeDates;lookbackDays]
    results:{[data;symbol;tradeDates;lookbackDays;dt]
        result:runAnalysis[data;symbol;dt;tradeDates;lookbackDays];
        $[null result; (); enlist (dt;result[1])]
    }[data;symbol;tradeDates;lookbackDays] each tradeDates;
    
    results:results where not null first each results;
    (!). flip results
    };

/ Export to CSV for further analysis
exportResults:{[allResults;filename]
    / Flatten results
    table:raze {[k;v]
        ([] 
            symbol:enlist first k;
            date:enlist last k;
            detectability:enlist v`detectability_score;
            weighted_correlation:enlist v`weighted_correlation
        )
    }' flip (key;value)@\:allResults;
    
    / Save to CSV
    `:$filename 0: csv 0: table;
    -1"Results exported to ",filename;
    };
Usage Examples
/ ============================================================================
/ QUICK START
/ ============================================================================

/ Run complete example
allResults:runExample[]

/ ============================================================================
/ PRODUCTION USAGE
/ ============================================================================

/ 1. Load your actual market data
data:select from tradeData where date within 2024.10.01 2024.11.01

/ 2. Define your trading days
tradeDates:`AAPL`MSFT!(2024.10.15 2024.10.18; 2024.10.16 2024.10.21)

/ 3. Run analysis for specific symbol and date
(metrics;score;merged):runAnalysis[data; `AAPL; 2024.10.15; tradeDates`AAPL; 21]

/ 4. Batch analysis
results:batchAnalysis[data; `AAPL; tradeDates`AAPL; 21]

/ 5. Real-time monitoring
/ During execution, periodically check:
liveData:select from streamingData where symbol=`AAPL, date=.z.d
baseline:loadBaseline[`AAPL; .z.d-1]
score:monitorRealTime[liveData; baseline; `AAPL]

/ 6. Export results
exportResults[allResults; "footprint_analysis_2024_11.csv"]

/ ============================================================================
/ INTEGRATION WITH EXISTING TCA SYSTEM
/ ============================================================================

/ Assuming you have a TCA database schema like:
/ trades: (symbol; date; timestamp; price; volume; side; ...)
/ market: (symbol; date; timestamp; bid; ask; bid_size; ask_size; ...)

/ Merge trade and market data
prepareData:{[trades;market]
    / Add trade direction to market data based on your executions
    data:aj[`symbol`date`timestamp; market; 
        select symbol, date, timestamp, myVolume:volume, mySide:side from trades];
    
    / Flag your trading activity
    data:update trade_direction:?[not null myVolume; mySide; 0N] from data;
    
    data
    };

/ Run daily footprint check
dailyFootprintCheck:{[date]
    / Get data for analysis window
    data:prepareData[trades; market] where date within (date-30; date);
    
    / Get symbols you traded today
    tradedSymbols:exec distinct symbol from trades where date=date;
    
    / Analyze each symbol
    results:{[data;date;sym]
        tradeDates:exec distinct date from trades where symbol=sym, date<=date;
        runAnalysis[data; sym; date; tradeDates; 21]
    }[data;date] each tradedSymbols;
    
    results
    };

/ Schedule this to run EOD
.z.ts:{[] dailyFootprintCheck[.z.d-1]}
\t 86400000  / Run daily