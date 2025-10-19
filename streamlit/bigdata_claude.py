import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# REQUIRED INSTALLATIONS:
# pip install streamlit pandas bigdata-client openai

# Set page config
st.set_page_config(page_title="Portfolio Risk & News Radar", layout="wide", page_icon="📊")

# API Configuration Section
st.sidebar.title("⚙️ API Configuration")
st.sidebar.markdown("---")

# Bigdata.com credentials
with st.sidebar.expander("🔐 Bigdata.com Credentials", expanded=False):
    bigdata_username = st.text_input("Username", value=os.environ.get("BIGDATA_USERNAME", ""), type="default")
    bigdata_password = st.text_input("Password", value=os.environ.get("BIGDATA_PASSWORD", ""), type="password")
    st.caption("Get credentials at [bigdata.com](https://bigdata.com)")

# API Connection Status
api_connected = False
if bigdata_username and bigdata_password:
    try:
        from bigdata_client import Bigdata
        bigdata = Bigdata(bigdata_username, bigdata_password)
        api_connected = True
        st.sidebar.success("✅ Connected to Bigdata.com")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")
        st.sidebar.caption("Install: `pip install bigdata-client`")
else:
    st.sidebar.warning("⚠️ Enter Bigdata.com credentials to enable real data")

# Helper functions for API integration
def resolve_company_entity(ticker, company_name):
    """Resolve company using Bigdata's knowledge graph"""
    if not api_connected:
        return {"ticker": ticker, "name": company_name, "id": None, "sector": "Unknown"}
    
    try:
        # Search for company in knowledge graph
        results = bigdata.knowledge_graph.companies.find_by_details(
            query=company_name,
            countries=["US"]
        )
        
        if results and len(results) > 0:
            company = results[0]
            return {
                "ticker": company.get("ticker", ticker),
                "name": company.get("name", company_name),
                "id": company.get("id"),
                "sector": company.get("sector", "Unknown"),
                "industry": company.get("industry", "Unknown")
            }
    except Exception as e:
        st.error(f"Entity resolution error: {str(e)}")
    
    return {"ticker": ticker, "name": company_name, "id": None, "sector": "Unknown"}

def search_company_news(company_info, days_back=30):
    """Search for company news using Bigdata API"""
    if not api_connected:
        return []
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build search query for risk-related content
        search_params = {
            "query": f"{company_info['name']} OR {company_info['ticker']}",
            "filters": {
                "entity": [company_info['id']] if company_info['id'] else [],
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "sources": ["news", "filings", "analyst_reports"],
                "sentiment": ["negative", "neutral"]  # Focus on risk signals
            },
            "scope": "rank-1",  # High quality sources
            "sort_by": "relevance",
            "limit": 50
        }
        
        # Execute search
        results = bigdata.search.query(**search_params)
        
        # Extract documents
        documents = []
        for doc in results.get("documents", []):
            documents.append({
                "date": doc.get("published_at"),
                "title": doc.get("title"),
                "source": doc.get("source_name"),
                "url": doc.get("url"),
                "snippet": doc.get("snippet"),
                "sentiment": doc.get("sentiment"),
                "full_text": doc.get("content", "")
            })
        
        return documents
        
    except Exception as e:
        st.error(f"Search error for {company_info['name']}: {str(e)}")
        return []

def classify_risk_with_claude(documents, company_name):
    """Use OpenAI to classify news into risk taxonomy"""
    from openai import OpenAI
    
    if not documents:
        return []
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("Set OPENAI_API_KEY environment variable for risk classification")
        return []
    
    client = OpenAI(api_key=api_key)
    
    # Prepare documents for analysis
    docs_text = "\n\n".join([
        f"Document {i+1}:\nDate: {doc['date']}\nTitle: {doc['title']}\nSource: {doc['source']}\nSnippet: {doc['snippet']}"
        for i, doc in enumerate(documents[:20])  # Limit to 20 docs
    ])
    
    prompt = f"""Analyze these news articles about {company_name} and classify each into risk categories.

Risk Categories:
- Regulatory: SEC investigations, compliance issues, new regulations
- Operational: Supply chain, production issues, management changes
- Financial: Earnings misses, debt concerns, cash flow problems
- Market: Competition, sector headwinds, market share loss
- Reputational: Lawsuits, scandals, negative publicity
- Strategic: M&A issues, business model changes

For each document, provide:
1. Risk type
2. Severity (Low/Medium/High/Critical)
3. Brief description
4. Document number reference

Documents:
{docs_text}

Respond with ONLY a JSON array like:
[
  {{
    "doc_index": 0,
    "risk_type": "Financial",
    "severity": "High",
    "description": "Missed Q3 earnings by 15%"
  }},
  ...
]"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        # Extract JSON from response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        classifications = json.loads(response_text)
        
        # Merge classifications with original documents
        classified_events = []
        for classification in classifications:
            doc_idx = classification.get("doc_index")
            if doc_idx < len(documents):
                doc = documents[doc_idx]
                classified_events.append({
                    "date": doc["date"],
                    "title": doc["title"],
                    "risk_type": classification["risk_type"],
                    "severity": classification["severity"],
                    "source": doc["source"],
                    "sentiment": doc.get("sentiment", "Negative"),
                    "url": doc["url"],
                    "description": classification["description"]
                })
        
        return classified_events
        
    except Exception as e:
        st.error(f"Risk classification error: {str(e)}")
        return []

def generate_factor_analysis_with_claude(company_name, risk_events):
    """Use OpenAI to analyze factor exposures from narrative"""
    from openai import OpenAI
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("Set OPENAI_API_KEY environment variable for factor analysis")
        return None
    
    client = OpenAI(api_key=api_key)
    
    events_text = "\n".join([
        f"- {e['date']}: {e['title']} ({e['risk_type']}, {e['severity']})"
        for e in risk_events[:15]
    ])
    
    prompt = f"""Based on these recent events for {company_name}, analyze the implied factor exposures:

Events:
{events_text}

Provide factor analysis for:
1. Value vs Growth tilt
2. Momentum (Positive/Negative/Neutral)
3. Quality score (High/Medium/Low)
4. Volatility expectation (High/Medium/Low)
5. Size factor characterization

Respond with ONLY a JSON object:
{{
  "value_growth": "Growth Tilt" or "Value Tilt" or "Neutral",
  "momentum": "Positive" or "Negative" or "Neutral",
  "quality": "High" or "Medium" or "Low",
  "volatility": "High" or "Medium" or "Low",
  "size_factor": "Large Cap" or "Mid Cap" or "Small Cap"
}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        return json.loads(response_text.strip())
        
    except Exception as e:
        st.error(f"Factor analysis error: {str(e)}")
        return None

def generate_hedge_recommendations_with_claude(company_name, risk_events, factor_analysis):
    """Generate hedge recommendations using OpenAI"""
    from openai import OpenAI
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    
    client = OpenAI(api_key=api_key)
    
    events_summary = "\n".join([
        f"- {e['severity']} {e['risk_type']}: {e['description']}"
        for e in risk_events[:10]
    ])
    
    prompt = f"""Given these risk events and factor profile for {company_name}, recommend hedging strategies:

Risk Events:
{events_summary}

Factor Profile:
{json.dumps(factor_analysis, indent=2)}

Provide 2-4 specific, actionable hedge recommendations. Consider:
- Put options for directional risk
- Sector inverse ETFs for systematic exposure
- Position sizing changes
- Pair trades

Respond with ONLY a JSON array:
[
  {{
    "strategy": "Strategy name",
    "description": "Specific actionable recommendation",
    "confidence": "High" or "Medium" or "Low"
  }},
  ...
]"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        return json.loads(response_text.strip())
        
    except Exception as e:
        st.error(f"Hedge recommendation error: {str(e)}")
        return []

# Sidebar - Portfolio Input
st.sidebar.title("📊 Portfolio Configuration")
st.sidebar.markdown("---")

# Session state for portfolio and analysis
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

# Portfolio input
st.sidebar.subheader("Add Holdings")
with st.sidebar.form("add_holding"):
    ticker = st.text_input("Ticker Symbol", placeholder="AAPL")
    company_name = st.text_input("Company Name", placeholder="Apple Inc.")
    shares = st.number_input("Shares", min_value=1, value=100)
    sector = st.selectbox("Sector", ["Technology", "Financials", "Healthcare", "Energy", "Consumer", "Industrial"])
    
    if st.form_submit_button("Add to Portfolio"):
        if ticker and company_name:
            st.session_state.portfolio.append({
                "ticker": ticker.upper(),
                "name": company_name,
                "shares": shares,
                "sector": sector
            })
            st.success(f"Added {ticker.upper()} to portfolio")
            st.rerun()

# Display current portfolio
st.sidebar.markdown("---")
st.sidebar.subheader("Current Portfolio")
for holding in st.session_state.portfolio:
    st.sidebar.text(f"{holding['ticker']}: {holding['shares']} shares")

if st.sidebar.button("Clear Portfolio"):
    st.session_state.portfolio = []
    st.session_state.analysis_cache = {}
    st.rerun()

# Main content
st.title("🎯 Portfolio Risk & News Radar")
st.markdown("Real-time risk monitoring powered by Bigdata.com API & OpenAI")

if not api_connected:
    st.warning("⚠️ Connect to Bigdata.com API in the sidebar to enable real-time analysis")

# Analysis controls
col1, col2, col3 = st.columns(3)
with col1:
    time_window_days = st.selectbox("Time Window", [7, 30, 90], format_func=lambda x: f"Last {x} Days")
with col2:
    min_severity = st.selectbox("Min Severity", ["All", "Medium", "High", "Critical"])
with col3:
    risk_filter = st.multiselect("Risk Types", 
                                  ["Regulatory", "Operational", "Financial", "Market", "Reputational", "Strategic"],
                                  default=["Regulatory", "Operational", "Financial", "Market", "Reputational", "Strategic"])

# Analyze button
if st.button("🔍 Run Analysis", type="primary", disabled=not api_connected):
    st.session_state.analysis_cache = {}
    
    with st.spinner("Analyzing portfolio..."):
        for holding in st.session_state.portfolio:
            # Resolve entity
            company_info = resolve_company_entity(holding["ticker"], holding["name"])
            
            # Search for news
            documents = search_company_news(company_info, days_back=time_window_days)
            
            # Classify risks
            risk_events = classify_risk_with_claude(documents, company_info["name"])
            
            # Factor analysis
            factor_analysis = generate_factor_analysis_with_claude(company_info["name"], risk_events)
            
            # Hedge recommendations
            hedge_recs = generate_hedge_recommendations_with_claude(
                company_info["name"], 
                risk_events, 
                factor_analysis
            )
            
            # Cache results
            st.session_state.analysis_cache[holding["ticker"]] = {
                "company_info": company_info,
                "risk_events": risk_events,
                "factor_analysis": factor_analysis,
                "hedge_recommendations": hedge_recs
            }
    
    st.success("✅ Analysis complete!")
    st.rerun()

st.markdown("---")

# Display analysis results
if len(st.session_state.portfolio) == 0:
    st.info("No holdings in portfolio. Add holdings using the sidebar.")
elif len(st.session_state.analysis_cache) == 0:
    st.info("Click 'Run Analysis' to analyze your portfolio with real-time data.")
else:
    # Portfolio overview tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Risk Dashboard", "🔍 Detailed Events", "💹 Factor Analysis", "🛡️ Hedge Recommendations"])
    
    with tab1:
        st.subheader("Portfolio Risk Overview")
        
        # Calculate aggregate risk scores
        portfolio_risks = []
        for holding in st.session_state.portfolio:
            if holding["ticker"] in st.session_state.analysis_cache:
                analysis = st.session_state.analysis_cache[holding["ticker"]]
                events = analysis["risk_events"]
                
                # Filter events
                filtered_events = [e for e in events if e["risk_type"] in risk_filter]
                if min_severity != "All":
                    severity_order = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
                    min_level = severity_order.get(min_severity, 0)
                    filtered_events = [e for e in filtered_events 
                                     if severity_order.get(e["severity"], 0) >= min_level]
                
                # Risk score calculation
                severity_weights = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
                risk_score = sum(severity_weights.get(e["severity"], 0) for e in filtered_events)
                
                portfolio_risks.append({
                    "Ticker": holding["ticker"],
                    "Company": analysis["company_info"]["name"],
                    "Sector": analysis["company_info"]["sector"],
                    "Events": len(filtered_events),
                    "Risk Score": risk_score,
                    "Highest Severity": max([e["severity"] for e in filtered_events], 
                                           key=lambda x: severity_weights.get(x, 0)) if filtered_events else "None"
                })
        
        if portfolio_risks:
            df_risks = pd.DataFrame(portfolio_risks)
            df_risks = df_risks.sort_values("Risk Score", ascending=False)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Holdings", len(st.session_state.portfolio))
            with col2:
                st.metric("High Risk Holdings", len(df_risks[df_risks["Risk Score"] >= 6]))
            with col3:
                total_events = df_risks["Events"].sum()
                st.metric("Total Risk Events", total_events)
            with col4:
                avg_risk = df_risks["Risk Score"].mean() if len(df_risks) > 0 else 0
                st.metric("Avg Risk Score", f"{avg_risk:.1f}")
            
            st.markdown("### Risk Heatmap")
            
            # Color code by risk score
            def color_risk_score(val):
                if val >= 9:
                    return 'background-color: #ff4444; color: white'
                elif val >= 6:
                    return 'background-color: #ffaa00; color: white'
                elif val >= 3:
                    return 'background-color: #ffdd00'
                else:
                    return 'background-color: #44ff44'
            
            styled_df = df_risks.style.applymap(color_risk_score, subset=['Risk Score'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Detailed Risk Events")
        
        # Select holding to view
        analyzed_tickers = [t for t in st.session_state.analysis_cache.keys()]
        if analyzed_tickers:
            selected_ticker = st.selectbox("Select Holding", analyzed_tickers)
            
            analysis = st.session_state.analysis_cache[selected_ticker]
            company_info = analysis["company_info"]
            events = analysis["risk_events"]
            
            st.markdown(f"### {company_info['name']} ({company_info['ticker']})")
            st.markdown(f"**Sector:** {company_info['sector']}")
            
            # Filter events
            filtered_events = [e for e in events if e["risk_type"] in risk_filter]
            
            st.markdown(f"**Found {len(filtered_events)} risk events**")
            
            # Display events
            for event in filtered_events:
                severity_colors = {
                    "Low": "🟢",
                    "Medium": "🟡",
                    "High": "🟠",
                    "Critical": "🔴"
                }
                
                with st.expander(f"{severity_colors[event['severity']]} {event['date'][:10] if event['date'] else 'N/A'} - {event['title']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Risk Type:** {event['risk_type']}")
                        st.markdown(f"**Severity:** {event['severity']}")
                    with col2:
                        st.markdown(f"**Source:** {event['source']}")
                        st.markdown(f"**Sentiment:** {event['sentiment']}")
                    with col3:
                        st.markdown(f"**Date:** {event['date'][:10] if event['date'] else 'N/A'}")
                        if event.get('url'):
                            st.markdown(f"[View Source]({event['url']})")
                    
                    st.markdown(f"**Analysis:** {event.get('description', 'N/A')}")
    
    with tab3:
        st.subheader("Factor Analysis & Exposures")
        
        factor_data = []
        for ticker, analysis in st.session_state.analysis_cache.items():
            company_info = analysis["company_info"]
            factors = analysis.get("factor_analysis") or {}
            factor_data.append({
                "Ticker": company_info["ticker"],
                "Company": company_info["name"],
                "Value/Growth": factors.get("value_growth", "N/A"),
                "Momentum": factors.get("momentum", "N/A"),
                "Quality": factors.get("quality", "N/A"),
                "Volatility": factors.get("volatility", "N/A"),
                "Size": factors.get("size_factor", "N/A")
            })
        
        if factor_data:
            df_factors = pd.DataFrame(factor_data)
            st.dataframe(df_factors, use_container_width=True, hide_index=True)
            
            st.markdown("### Portfolio-Level Factor Exposures")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Value vs Growth**")
                value_count = len([f for f in factor_data if f["Value/Growth"] == "Value Tilt"])
                growth_count = len([f for f in factor_data if f["Value/Growth"] == "Growth Tilt"])
                total = len(factor_data)
                if total > 0:
                    st.progress(growth_count / total)
                    st.caption(f"{growth_count} Growth, {value_count} Value")
                
                st.markdown("**Quality Distribution**")
                high_quality = len([f for f in factor_data if f["Quality"] == "High"])
                st.metric("High Quality Holdings", f"{high_quality}/{total}")
            
            with col2:
                st.markdown("**Momentum Profile**")
                positive_momentum = len([f for f in factor_data if f["Momentum"] == "Positive"])
                if total > 0:
                    st.progress(positive_momentum / total)
                    st.caption(f"{positive_momentum} Positive Momentum")
                
                st.markdown("**Volatility Exposure**")
                high_vol = len([f for f in factor_data if f["Volatility"] == "High"])
                st.metric("High Volatility Holdings", f"{high_vol}/{total}")
    
    with tab4:
        st.subheader("🛡️ Factor-Aware Hedge Recommendations")
        
        for ticker, analysis in st.session_state.analysis_cache.items():
            company_info = analysis["company_info"]
            recommendations = analysis.get("hedge_recommendations") or []
            
            st.markdown(f"### {company_info['name']} ({company_info['ticker']})")
            
            if recommendations:
                for rec in recommendations:
                    confidence_colors = {
                        "High": "🟢",
                        "Medium": "🟡",
                        "Low": "🔴"
                    }
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**Strategy:** {rec['strategy']}")
                        st.markdown(f"**Confidence:** {confidence_colors.get(rec['confidence'], '⚪')} {rec['confidence']}")
                    with col2:
                        st.info(rec['description'])
            else:
                st.info("No specific hedge recommendations at this time. Continue monitoring.")
            
            st.markdown("---")

# Export functionality
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Export")

if st.sidebar.button("Generate Digest", disabled=len(st.session_state.analysis_cache) == 0):
    # Generate comprehensive digest
    digest = {
        "generated_at": datetime.now().isoformat(),
        "portfolio_summary": {
            "total_holdings": len(st.session_state.portfolio),
            "holdings": st.session_state.portfolio
        },
        "analysis_results": st.session_state.analysis_cache
    }
    
    # Display download button
    st.sidebar.download_button(
        label="Download JSON Digest",
        data=json.dumps(digest, indent=2),
        file_name=f"portfolio_risk_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.sidebar.success("Digest generated!")

# Footer
st.markdown("---")
st.caption("Portfolio Risk & News Radar | Powered by Bigdata.com API + OpenAI")
