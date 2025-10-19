import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import List, Dict

# Page config
st.set_page_config(page_title="Pre-Open Gap Triager", layout="wide", initial_sidebar_state="collapsed")

# ============================================================================
# CONFIGURATION - Replace with your real data sources
# ============================================================================

# Mock watchlist - replace with your broker feed (IB, Bloomberg, etc.)
def get_watchlist_with_gaps():
    """Replace this with your actual price feed API"""
    return pd.DataFrame({
        'ticker': ['AAPL', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN'],
        'prev_close': [178.50, 142.30, 248.50, 485.20, 168.90, 178.30],
        'premarket_price': [182.40, 147.80, 243.10, 492.60, 171.20, 179.10],
        'volume': [450000, 890000, 1200000, 320000, 180000, 210000],
    })

def calculate_gaps(df):
    """Calculate gap % and filter for significant moves"""
    df['gap_pct'] = ((df['premarket_price'] - df['prev_close']) / df['prev_close']) * 100
    df['gap_abs'] = abs(df['gap_pct'])
    return df[df['gap_abs'] >= 2.0].sort_values('gap_abs', ascending=False)

# ============================================================================
# BIGDATA API INTEGRATION - Plug in your API here
# ============================================================================

def fetch_catalysts_from_bigdata(ticker: str, hours_back: int = 24) -> List[Dict]:
    """
    Replace this with actual BigData API call
    Docs: https://docs.bigdata.com/how-to-guides/introduction
    
    Query structure from your spec:
    Entity(company) & (Topic(Guidance | M&A | Litigation | ExecChange | Regulation) 
    | Similarity("price sensitive" OR "guidance" OR "restructuring"))
    """
    
    # MOCK DATA - Replace with real API call
    # Real implementation would look like:
    # response = requests.post(
    #     "https://api.bigdata.com/v1/search",
    #     headers={"Authorization": f"Bearer {YOUR_API_KEY}"},
    #     json={
    #         "query": f'Entity({ticker}) & (Topic(Guidance | M&A | Litigation) | Similarity("price sensitive"))',
    #         "time_range": {"hours": hours_back},
    #         "doc_types": ["News", "PressRelease", "Transcript"],
    #         "source_ranks": [1, 2],
    #         "retention": "long"
    #     }
    # )
    
    mock_catalysts = {
        'AAPL': [
            {
                'headline': 'Apple announces Q4 earnings beat, raises FY guidance by 5%',
                'doc_type': 'PressRelease',
                'source': 'Apple IR',
                'source_rank': 1,
                'timestamp': datetime.now() - timedelta(hours=2),
                'sentiment': 'positive',
                'excerpt': 'Revenue of $89.5B vs $88.2B expected. iPhone revenue up 8% YoY. Management raised full-year guidance...',
                'url': 'https://investor.apple.com/news/default.aspx'
            },
            {
                'headline': 'Morgan Stanley upgrades AAPL to Overweight on AI momentum',
                'doc_type': 'News',
                'source': 'Bloomberg',
                'source_rank': 1,
                'timestamp': datetime.now() - timedelta(hours=4),
                'sentiment': 'positive',
                'excerpt': 'Analyst Katy Huberty raised price target to $220 citing strong iPhone 16 pre-orders...',
                'url': 'https://bloomberg.com'
            }
        ],
        'NVDA': [
            {
                'headline': 'Nvidia announces new AI chip delays due to design flaw',
                'doc_type': 'News',
                'source': 'Reuters',
                'source_rank': 1,
                'timestamp': datetime.now() - timedelta(hours=1),
                'sentiment': 'negative',
                'excerpt': 'The Blackwell B200 chips will ship 3 months later than expected. NVDA cited "design modifications"...',
                'url': 'https://reuters.com'
            }
        ],
        'TSLA': [
            {
                'headline': 'Tesla misses delivery targets, cites factory retooling',
                'doc_type': 'PressRelease',
                'source': 'Tesla IR',
                'source_rank': 1,
                'timestamp': datetime.now() - timedelta(hours=3),
                'sentiment': 'negative',
                'excerpt': 'Q3 deliveries of 435K vs 455K expected. Company states production paused for Cybertruck ramp...',
                'url': 'https://ir.tesla.com'
            }
        ],
        'META': [
            {
                'headline': 'No significant catalyst found in last 24h',
                'doc_type': 'System',
                'source': 'BigData',
                'source_rank': 99,
                'timestamp': datetime.now(),
                'sentiment': 'neutral',
                'excerpt': 'Possible sector rotation or broad market move. Check macro conditions.',
                'url': ''
            }
        ]
    }
    
    return mock_catalysts.get(ticker, [])

def score_catalyst(doc: Dict) -> float:
    """Score catalysts based on source rank, freshness, doc type"""
    score = 0.0
    
    # Source rank weight (higher rank = lower score)
    if doc['source_rank'] == 1:
        score += 10.0
    elif doc['source_rank'] == 2:
        score += 7.0
    else:
        score += 3.0
    
    # Freshness weight (exponential decay)
    hours_old = (datetime.now() - doc['timestamp']).total_seconds() / 3600
    score += max(0, 10.0 - hours_old)
    
    # Doc type weight
    if doc['doc_type'] == 'PressRelease':
        score += 8.0
    elif doc['doc_type'] == 'Transcript':
        score += 7.0
    elif doc['doc_type'] == 'News':
        score += 5.0
    
    return round(score, 1)

def determine_playbook(doc: Dict, gap_pct: float) -> Dict:
    """Generate trading playbook based on catalyst + price action"""
    
    if doc['doc_type'] == 'System':  # No catalyst found
        return {
            'strategy': 'Mean Revert',
            'confidence': 'Low',
            'rationale': 'No primary catalyst—likely sector rotation or rumor',
            'sizing': 'Small position, tight stops'
        }
    
    sentiment = doc['sentiment']
    doc_type = doc['doc_type']
    source_rank = doc['source_rank']
    
    # Strong catalyst + aligned price action
    if source_rank <= 1 and abs(gap_pct) > 3:
        if (sentiment == 'positive' and gap_pct > 0) or (sentiment == 'negative' and gap_pct < 0):
            return {
                'strategy': 'Momentum Continuation',
                'confidence': 'High',
                'rationale': 'Primary source + strong gap alignment',
                'sizing': 'Standard size, trail stops'
            }
    
    # Weak catalyst or misaligned
    if source_rank >= 2 or doc_type == 'News':
        return {
            'strategy': 'Fade / Mean Revert',
            'confidence': 'Medium',
            'rationale': 'Secondary source or rumor—watch for reversal',
            'sizing': 'Reduced size, wide stops'
        }
    
    return {
        'strategy': 'Wait & Watch',
        'confidence': 'Medium',
        'rationale': 'Mixed signals—let price confirm',
        'sizing': 'Paper trade or minimal'
    }

def suggest_hedge(ticker: str, gap_pct: float, sector: str = 'Tech') -> Dict:
    """Suggest hedge instruments for beta-neutral positioning"""
    
    # Sector/index mapping
    hedges = {
        'Tech': {'instrument': 'NQ (Nasdaq Futures)', 'beta': 1.2},
        'Broad': {'instrument': 'ES (S&P Futures)', 'beta': 1.0},
        'Europe': {'instrument': 'SX5E (Euro Stoxx)', 'beta': 0.9}
    }
    
    hedge = hedges.get(sector, hedges['Broad'])
    
    # Calculate beta-neutral size (simplified)
    notional = 10000  # Mock $10k position
    hedge_size = int((notional * hedge['beta']) / 50)  # Assuming $50/point futures
    
    return {
        'instrument': hedge['instrument'],
        'action': 'Short' if gap_pct > 0 else 'Long',
        'size': f"{hedge_size} contracts",
        'beta': hedge['beta']
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎯 Pre-Open Gap & Catalyst Triager")
st.caption(f"UK Market Hours | Data as of {datetime.now().strftime('%H:%M:%S')}")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    gap_threshold = st.slider("Gap Threshold (%)", 1.0, 5.0, 2.0, 0.5)
    hours_lookback = st.slider("Catalyst Lookback (hours)", 6, 48, 24, 6)
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Get data
df = get_watchlist_with_gaps()
df_gaps = calculate_gaps(df)

if df_gaps.empty:
    st.info(f"No tickers with gaps ≥{gap_threshold}% in pre-market")
    st.stop()

# Add stance column
def get_stance(ticker):
    catalysts = fetch_catalysts_from_bigdata(ticker, hours_lookback)
    if catalysts:
        return catalysts[0]['sentiment'].upper()
    return 'NEUTRAL'

df_gaps['stance'] = df_gaps['ticker'].apply(get_stance)

# Layout: Left table + Right detail panel
col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Watchlist Gaps")
    
    # Format table for display
    display_df = df_gaps[['ticker', 'gap_pct', 'stance', 'volume']].copy()
    display_df['gap_pct'] = display_df['gap_pct'].apply(lambda x: f"{x:+.2f}%")
    display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,}")
    display_df.columns = ['Ticker', 'Gap %', 'Stance', 'Vol']
    
    # Interactive table selection
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Get selected ticker
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_ticker = df_gaps.iloc[selected_idx]['ticker']
    else:
        selected_ticker = df_gaps.iloc[0]['ticker']

with col_right:
    st.subheader(f"📊 {selected_ticker} Analysis")
    
    # Get ticker data
    ticker_data = df_gaps[df_gaps['ticker'] == selected_ticker].iloc[0]
    gap_pct = ticker_data['gap_pct']
    
    # Display price action
    col1, col2, col3 = st.columns(3)
    col1.metric("Prev Close", f"${ticker_data['prev_close']:.2f}")
    col2.metric("Pre-Market", f"${ticker_data['premarket_price']:.2f}")
    col3.metric("Gap", f"{gap_pct:+.2f}%", delta=f"${ticker_data['premarket_price'] - ticker_data['prev_close']:.2f}")
    
    st.divider()
    
    # Fetch and display catalysts
    catalysts = fetch_catalysts_from_bigdata(selected_ticker, hours_lookback)
    
    if catalysts:
        # Score and sort
        for cat in catalysts:
            cat['score'] = score_catalyst(cat)
        catalysts = sorted(catalysts, key=lambda x: x['score'], reverse=True)[:3]
        
        st.markdown("### 🔍 Top Catalysts")
        
        for i, cat in enumerate(catalysts, 1):
            with st.expander(f"**{i}. {cat['headline']}**", expanded=(i==1)):
                col_a, col_b, col_c = st.columns([2, 1, 1])
                col_a.caption(f"Source: {cat['source']} ({cat['doc_type']})")
                col_b.caption(f"Score: {cat['score']}")
                col_c.caption(f"{int((datetime.now() - cat['timestamp']).total_seconds() / 3600)}h ago")
                
                st.markdown(cat['excerpt'])
                
                if cat['url']:
                    st.link_button("📄 Read Full Source", cat['url'], use_container_width=True)
        
        # Playbook
        st.divider()
        st.markdown("### 🎲 Suggested Playbook")
        
        playbook = determine_playbook(catalysts[0], gap_pct)
        
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Strategy", playbook['strategy'])
        col_p2.metric("Confidence", playbook['confidence'])
        
        st.info(f"**Rationale:** {playbook['rationale']}\n\n**Sizing:** {playbook['sizing']}")
    
    else:
        st.warning("No catalysts found in lookback period")

# Footer: Hedge suggestions
st.divider()
st.subheader("🛡️ Hedge Helper")

hedge = suggest_hedge(selected_ticker, gap_pct)

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
col_h1.metric("Instrument", hedge['instrument'])
col_h2.metric("Action", hedge['action'])
col_h3.metric("Size", hedge['size'])
col_h4.metric("Beta", hedge['beta'])

st.caption("Beta-neutral hedge calculation based on simplified 20-day correlation. Adjust for your risk model.")

# Quick copy
hedge_order = f"{hedge['action']} {hedge['size']} {hedge['instrument']}"
st.code(hedge_order, language=None)

# Footer note
st.divider()
st.caption("⚠️ This is an MVP demo with mock data. Replace API calls with your live feeds (BigData, IB, Bloomberg, etc.)")
