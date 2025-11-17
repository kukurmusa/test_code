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

/ Schedule this to run EOD
.z.ts:{[] dailyFootprintCheck[.z.d-1]}
\t 86400000  / Run daily
