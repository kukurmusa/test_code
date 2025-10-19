import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import time, timedelta

# ============================================================================
# NEWS ALPHA TESTING FOR EXECUTION ALGORITHMS
# Tests if news sentiment can improve VWAP and Arrival Price algo performance
# ============================================================================

class ExecutionAlgoNewsTester:
    """
    Tests whether news sentiment provides actionable alpha for:
    1. VWAP algos (schedule optimization: front-load vs back-load)
    2. Arrival price algos (urgency/aggressiveness optimization)
    """
    
    def __init__(self, news_df, intraday_price_df):
        """
        Parameters:
        -----------
        news_df : DataFrame with ['ticker', 'timestamp', 'sentiment']
            News events (typically pre-market or early intraday)
        
        intraday_price_df : DataFrame with ['ticker', 'timestamp', 'price', 'volume']
            Intraday price/volume data (e.g., 1-min or 5-min bars)
        """
        self.news_df = news_df.copy()
        self.price_df = intraday_price_df.copy()
        
        # Ensure datetime
        self.news_df['timestamp'] = pd.to_datetime(self.news_df['timestamp'])
        self.price_df['timestamp'] = pd.to_datetime(self.price_df['timestamp'])
        
        # Add date column
        self.news_df['date'] = self.news_df['timestamp'].dt.date
        self.price_df['date'] = self.price_df['timestamp'].dt.date
        
        self.results = {}
    
    def calculate_intraday_vwap(self, start_time=time(9, 30), end_time=time(16, 0)):
        """Calculate actual VWAP for each ticker-date"""
        # Filter to market hours
        mask = (self.price_df['timestamp'].dt.time >= start_time) & \
               (self.price_df['timestamp'].dt.time <= end_time)
        df = self.price_df[mask].copy()
        
        # Calculate VWAP by ticker-date
        df['dollar_volume'] = df['price'] * df['volume']
        vwap = df.groupby(['ticker', 'date']).agg({
            'dollar_volume': 'sum',
            'volume': 'sum'
        })
        vwap['vwap'] = vwap['dollar_volume'] / vwap['volume']
        
        return vwap[['vwap']].reset_index()
    
    def calculate_intraday_trajectory(self, start_time=time(9, 30), end_time=time(16, 0)):
        """
        Calculate how price evolves throughout the day
        Returns: slope of intraday price path (positive = uptrend, negative = downtrend)
        """
        mask = (self.price_df['timestamp'].dt.time >= start_time) & \
               (self.price_df['timestamp'].dt.time <= end_time)
        df = self.price_df[mask].copy()
        
        # Add minutes since open
        df['minutes_since_open'] = (df['timestamp'].dt.hour - start_time.hour) * 60 + \
                                    (df['timestamp'].dt.minute - start_time.minute)
        
        # Calculate trend for each ticker-date
        trajectory_results = []
        for (ticker, date), group in df.groupby(['ticker', 'date']):
            if len(group) < 10:  # Need enough points
                continue
            
            # Normalize prices to percentage from open
            open_price = group.iloc[0]['price']
            group['pct_from_open'] = (group['price'] / open_price - 1) * 100
            
            # Linear regression: price ~ time
            X = group['minutes_since_open'].values
            y = group['pct_from_open'].values
            
            if len(X) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                
                # Also calculate early vs late performance
                midpoint = len(group) // 2
                early_return = (group.iloc[midpoint]['price'] / open_price - 1)
                late_return = (group.iloc[-1]['price'] / group.iloc[midpoint]['price'] - 1)
                full_return = (group.iloc[-1]['price'] / open_price - 1)
                
                trajectory_results.append({
                    'ticker': ticker,
                    'date': date,
                    'open_price': open_price,
                    'close_price': group.iloc[-1]['price'],
                    'full_day_return': full_return,
                    'trajectory_slope': slope,  # % per minute
                    'trajectory_r2': r_value**2,
                    'early_return': early_return,  # First half of day
                    'late_return': late_return,    # Second half of day
                    'trend_strength': abs(slope) * r_value**2  # Slope weighted by fit
                })
        
        return pd.DataFrame(trajectory_results)
    
    def test_vwap_schedule_optimization(self):
        """
        Test if news sentiment predicts whether to front-load or back-load VWAP execution
        
        Key insight:
        - Positive news + upward trajectory → front-load (buy early, price going up)
        - Negative news + downward trajectory → back-load (buy late, price going down)
        """
        print("\n" + "="*70)
        print("VWAP SCHEDULE OPTIMIZATION TEST")
        print("="*70)
        
        # Get intraday trajectories
        trajectories = self.calculate_intraday_trajectory()
        
        # Merge with news (use morning news before 10:30 AM)
        morning_news = self.news_df[self.news_df['timestamp'].dt.time <= time(10, 30)].copy()
        morning_news = morning_news.groupby(['ticker', 'date'])['sentiment'].mean().reset_index()
        
        merged = trajectories.merge(morning_news, on=['ticker', 'date'], how='inner')
        
        if len(merged) < 10:
            print("Insufficient data for analysis")
            return None
        
        # Test 1: Does sentiment predict trajectory slope?
        corr_slope = merged[['sentiment', 'trajectory_slope']].corr().iloc[0, 1]
        
        print(f"\n1. Sentiment vs Intraday Trajectory")
        print(f"   Correlation: {corr_slope:.4f}")
        print(f"   Sample size: {len(merged)}")
        
        # Test 2: Does sentiment predict early vs late returns?
        corr_early = merged[['sentiment', 'early_return']].corr().iloc[0, 1]
        corr_late = merged[['sentiment', 'late_return']].corr().iloc[0, 1]
        
        print(f"\n2. Sentiment vs Early/Late Returns")
        print(f"   Early return correlation: {corr_early:.4f}")
        print(f"   Late return correlation: {corr_late:.4f}")
        
        # Test 3: Quantile analysis - schedule optimization potential
        merged['sentiment_quintile'] = pd.qcut(merged['sentiment'], q=5, 
                                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                duplicates='drop')
        
        print(f"\n3. Optimal Schedule by Sentiment Quintile")
        print(f"   {'Quintile':<10} {'Early Ret':<12} {'Late Ret':<12} {'Recommendation'}")
        print(f"   {'-'*60}")
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = merged[merged['sentiment_quintile'] == q]
            if len(q_data) > 0:
                early_ret = q_data['early_return'].mean()
                late_ret = q_data['late_return'].mean()
                
                # Recommendation logic
                if early_ret > late_ret + 0.001:  # 10 bps threshold
                    rec = "FRONT-LOAD ⬆"
                elif late_ret > early_ret + 0.001:
                    rec = "BACK-LOAD ⬇"
                else:
                    rec = "NEUTRAL →"
                
                print(f"   {q:<10} {early_ret:>10.2%} {late_ret:>10.2%}   {rec}")
        
        # Test 4: Simulated VWAP performance
        print(f"\n4. Simulated VWAP Slippage (BUY orders)")
        
        # Simulate different strategies
        for strategy_name, weight_func in [
            ("Standard VWAP", lambda x: 1.0),  # Uniform
            ("Front-Load (High Sentiment)", lambda x: 2.0 if x > 0 else 0.5),
            ("Back-Load (Low Sentiment)", lambda x: 0.5 if x < 0 else 2.0)
        ]:
            slippages = []
            for _, row in merged.iterrows():
                sentiment = row['sentiment']
                early_ret = row['early_return']
                late_ret = row['late_return']
                
                # Weight for front-loading (0-1, where 1 = all early)
                if strategy_name == "Standard VWAP":
                    front_weight = 0.5
                elif strategy_name.startswith("Front-Load"):
                    front_weight = 0.7 if sentiment > 0 else 0.3
                else:  # Back-load
                    front_weight = 0.3 if sentiment < 0 else 0.7
                
                # Weighted execution price vs VWAP (assuming VWAP is midpoint)
                # Positive early_ret means buying early is expensive
                exec_slippage = front_weight * early_ret + (1 - front_weight) * late_ret
                slippages.append(exec_slippage)
            
            avg_slippage = np.mean(slippages) * 10000  # in basis points
            print(f"   {strategy_name:<35} {avg_slippage:>8.1f} bps")
        
        self.results['vwap_test'] = merged
        return merged
    
    def test_arrival_price_urgency(self):
        """
        Test if news sentiment predicts optimal urgency for Arrival Price algo
        
        Key insight:
        - Strong positive news → high urgency (price will run away, be aggressive)
        - Weak/negative news → low urgency (price may revert, be patient)
        """
        print("\n" + "="*70)
        print("ARRIVAL PRICE URGENCY OPTIMIZATION TEST")
        print("="*70)
        
        # Get intraday trajectories
        trajectories = self.calculate_intraday_trajectory()
        
        # Merge with news
        morning_news = self.news_df[self.news_df['timestamp'].dt.time <= time(10, 30)].copy()
        morning_news = morning_news.groupby(['ticker', 'date'])['sentiment'].mean().reset_index()
        
        merged = trajectories.merge(morning_news, on=['ticker', 'date'], how='inner')
        
        if len(merged) < 10:
            print("Insufficient data for analysis")
            return None
        
        # Test 1: Does sentiment predict momentum continuation?
        print(f"\n1. Sentiment vs Price Momentum")
        corr_momentum = merged[['sentiment', 'full_day_return']].corr().iloc[0, 1]
        print(f"   Correlation: {corr_momentum:.4f}")
        
        # Test 2: Does strong sentiment predict trend persistence?
        merged['abs_sentiment'] = merged['sentiment'].abs()
        corr_strength = merged[['abs_sentiment', 'trend_strength']].corr().iloc[0, 1]
        print(f"\n2. Sentiment Magnitude vs Trend Strength")
        print(f"   Correlation: {corr_strength:.4f}")
        print(f"   (Higher = stronger sentiment predicts more persistent trends)")
        
        # Test 3: Quantile analysis for urgency recommendations
        merged['sentiment_quintile'] = pd.qcut(merged['sentiment'], q=5, 
                                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                duplicates='drop')
        
        print(f"\n3. Urgency Recommendations by Sentiment")
        print(f"   {'Quintile':<10} {'Avg Return':<12} {'Trend Str':<12} {'Urgency'}")
        print(f"   {'-'*65}")
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = merged[merged['sentiment_quintile'] == q]
            if len(q_data) > 0:
                avg_ret = q_data['full_day_return'].mean()
                trend_str = q_data['trend_strength'].mean()
                
                # Urgency recommendation
                if q in ['Q1']:
                    urgency = "LOW (patient)"
                elif q in ['Q2', 'Q3', 'Q4']:
                    urgency = "MEDIUM"
                else:
                    urgency = "HIGH (aggressive)"
                
                print(f"   {q:<10} {avg_ret:>10.2%} {trend_str:>10.4f}   {urgency}")
        
        # Test 4: Implementation shortfall simulation
        print(f"\n4. Simulated Implementation Shortfall (BUY orders)")
        print(f"   (vs arrival price = open price)")
        
        for strategy_name, participation_func in [
            ("Standard (50% POV)", lambda x: 0.5),
            ("News-Adjusted POV", lambda x: 0.7 if x > 0.5 else (0.3 if x < -0.5 else 0.5)),
            ("Conservative (30% POV)", lambda x: 0.3)
        ]:
            shortfalls = []
            for _, row in merged.iterrows():
                sentiment = row['sentiment']
                full_return = row['full_day_return']
                
                # Higher participation = execute faster = less drift but more impact
                if strategy_name == "Standard (50% POV)":
                    participation = 0.5
                elif strategy_name == "News-Adjusted POV":
                    if sentiment > 0.5:
                        participation = 0.7  # Aggressive on positive news
                    elif sentiment < -0.5:
                        participation = 0.3  # Patient on negative news
                    else:
                        participation = 0.5
                else:
                    participation = 0.3
                
                # Simplified model:
                # - Fast execution (high POV) = less drift but more impact
                # - Slow execution (low POV) = more drift but less impact
                execution_time_pct = 1 - participation  # 0=instant, 1=all day
                price_drift = full_return * execution_time_pct
                market_impact = 0.0010 * participation  # 10 bps max impact
                
                shortfall = price_drift + market_impact
                shortfalls.append(shortfall)
            
            avg_shortfall = np.mean(shortfalls) * 10000  # basis points
            print(f"   {strategy_name:<25} {avg_shortfall:>8.1f} bps")
        
        self.results['arrival_test'] = merged
        return merged
    
    def test_news_timing_alpha(self):
        """
        Test if NEWS TIMING matters - do news right at market open have more impact?
        """
        print("\n" + "="*70)
        print("NEWS TIMING IMPACT TEST")
        print("="*70)
        
        trajectories = self.calculate_intraday_trajectory()
        
        # Categorize news by timing
        news_with_timing = self.news_df.copy()
        news_with_timing['hour'] = news_with_timing['timestamp'].dt.hour
        news_with_timing['news_category'] = pd.cut(
            news_with_timing['hour'],
            bins=[0, 9, 10, 12, 16, 24],
            labels=['Pre-Market', 'Market Open', 'Mid-Morning', 'Afternoon', 'After-Hours']
        )
        
        # Test each timing category
        for category in ['Pre-Market', 'Market Open', 'Mid-Morning', 'Afternoon']:
            cat_news = news_with_timing[news_with_timing['news_category'] == category]
            if len(cat_news) < 5:
                continue
            
            cat_news_agg = cat_news.groupby(['ticker', 'date'])['sentiment'].mean().reset_index()
            merged = trajectories.merge(cat_news_agg, on=['ticker', 'date'], how='inner')
            
            if len(merged) > 5:
                corr = merged[['sentiment', 'full_day_return']].corr().iloc[0, 1]
                print(f"   {category:<15} Correlation: {corr:>7.4f}  (n={len(merged)})")
    
    def plot_vwap_results(self):
        """Visualize VWAP optimization results"""
        if 'vwap_test' not in self.results:
            print("Run test_vwap_schedule_optimization() first")
            return
        
        df = self.results['vwap_test']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sentiment vs Trajectory Slope
        axes[0, 0].scatter(df['sentiment'], df['trajectory_slope'], alpha=0.6)
        axes[0, 0].set_xlabel('Morning Sentiment')
        axes[0, 0].set_ylabel('Intraday Trajectory Slope (% per min)')
        axes[0, 0].set_title('Sentiment vs Price Trajectory')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.3)
        
        # 2. Early vs Late Returns by Sentiment
        df_sorted = df.sort_values('sentiment')
        df_sorted['rolling_early'] = df_sorted['early_return'].rolling(window=20, center=True).mean()
        df_sorted['rolling_late'] = df_sorted['late_return'].rolling(window=20, center=True).mean()
        
        axes[0, 1].plot(df_sorted['sentiment'], df_sorted['rolling_early'], 
                       label='Early Return (1st half)', linewidth=2)
        axes[0, 1].plot(df_sorted['sentiment'], df_sorted['rolling_late'], 
                       label='Late Return (2nd half)', linewidth=2)
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Average Return')
        axes[0, 1].set_title('Optimal Execution Timing')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # 3. Quantile Returns
        quintile_stats = df.groupby('sentiment_quintile').agg({
            'early_return': 'mean',
            'late_return': 'mean'
        })
        
        x = np.arange(len(quintile_stats))
        width = 0.35
        axes[1, 0].bar(x - width/2, quintile_stats['early_return'], width, 
                      label='Early Return', alpha=0.8)
        axes[1, 0].bar(x + width/2, quintile_stats['late_return'], width,
                      label='Late Return', alpha=0.8)
        axes[1, 0].set_xlabel('Sentiment Quintile')
        axes[1, 0].set_ylabel('Average Return')
        axes[1, 0].set_title('Early vs Late Returns by Sentiment')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(quintile_stats.index)
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # 4. Distribution of trajectory slopes
        positive_sent = df[df['sentiment'] > 0]['trajectory_slope']
        negative_sent = df[df['sentiment'] < 0]['trajectory_slope']
        
        axes[1, 1].hist(positive_sent, bins=30, alpha=0.5, label='Positive Sentiment', density=True)
        axes[1, 1].hist(negative_sent, bins=30, alpha=0.5, label='Negative Sentiment', density=True)
        axes[1, 1].set_xlabel('Intraday Trajectory Slope')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Price Trajectory Distribution')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Sample news data (morning news)
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    news_data = []
    for date in dates:
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            # Morning news between 8-10 AM
            news_time = pd.Timestamp.combine(date, time(9, 0)) + \
                       pd.Timedelta(minutes=np.random.randint(0, 120))
            news_data.append({
                'ticker': ticker,
                'timestamp': news_time,
                'sentiment': np.random.randn()  # Sentiment score
            })
    news_df = pd.DataFrame(news_data)
    
    # Sample intraday price data (5-min bars from 9:30-16:00)
    price_data = []
    for date in dates:
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            # Simulate intraday prices with some trend based on sentiment
            day_sentiment = news_df[(news_df['ticker'] == ticker) & 
                                   (news_df['timestamp'].dt.date == date.date())]['sentiment'].mean()
            
            current_time = pd.Timestamp.combine(date, time(9, 30))
            end_time = pd.Timestamp.combine(date, time(16, 0))
            base_price = 100
            
            while current_time <= end_time:
                # Add trend based on sentiment
                minutes_elapsed = (current_time.hour - 9.5) * 60 + current_time.minute
                trend = day_sentiment * 0.001 * minutes_elapsed  # Sentiment-driven trend
                noise = np.random.randn() * 0.1
                
                price_data.append({
                    'ticker': ticker,
                    'timestamp': current_time,
                    'price': base_price * (1 + trend + noise),
                    'volume': np.random.randint(1000, 5000)
                })
                
                current_time += pd.Timedelta(minutes=5)
    
    price_df = pd.DataFrame(price_data)
    
    # Run tests
    tester = ExecutionAlgoNewsTester(news_df, price_df)
    
    print("\n" + "="*70)
    print("EXECUTION ALGO NEWS ALPHA TESTING")
    print("="*70)
    
    # Test 1: VWAP optimization
    tester.test_vwap_schedule_optimization()
    
    # Test 2: Arrival price urgency
    tester.test_arrival_price_urgency()
    
    # Test 3: News timing
    tester.test_news_timing_alpha()
    
    # Visualize
    tester.plot_vwap_results()
    plt.show()
