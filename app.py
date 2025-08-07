"""
Minimal Streamlit Cloud Entry Point
Twitter Sentiment Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1DA1F2;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    
    # Tweet count slider
    tweet_count = st.slider(
        "Number of tweets to analyze",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    # Analysis mode
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Real-time Demo", "Batch Analysis", "Historical Data"]
    )
    
    # Refresh button
    if st.button("🔄 Refresh Analysis", type="primary"):
        st.rerun()

# Generate demo data
@st.cache_data
def generate_sentiment_data(num_tweets):
    """Generate realistic demo sentiment data"""
    sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    topics = ['technology', 'sports', 'politics', 'entertainment', 'business', 'health', 'environment']
    
    data = []
    for i in range(num_tweets):
        sentiment = random.choices(
            sentiments, 
            weights=[0.4, 0.3, 0.3]  # Slightly more positive
        )[0]
        
        confidence = random.uniform(0.65, 0.98)
        topic = random.choice(topics)
        
        # Generate realistic tweet text
        sample_texts = {
            'POSITIVE': [
                f"Amazing breakthrough in {topic}! This is fantastic news 🎉",
                f"Really excited about the latest developments in {topic}",
                f"Great to see positive changes in {topic} industry",
                f"Love the innovation happening in {topic} space! 💖"
            ],
            'NEGATIVE': [
                f"Disappointed with the recent {topic} situation 😞",
                f"This {topic} issue needs urgent attention",
                f"Not happy with how {topic} is being handled",
                f"Concerned about the direction of {topic} lately"
            ],
            'NEUTRAL': [
                f"Here's an update on the {topic} situation",
                f"New report shows {topic} statistics for this quarter",
                f"Analysis: What the {topic} data really means",
                f"Breaking: {topic} announcement scheduled for next week"
            ]
        }
        
        tweet_text = random.choice(sample_texts[sentiment])
        
        data.append({
            'id': f"tweet_{i+1}",
            'text': tweet_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'topic': topic,
            'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 1440)),
            'retweets': random.randint(0, 1000),
            'likes': random.randint(0, 5000)
        })
    
    return pd.DataFrame(data)

# Generate data
with st.spinner(f"🔄 Analyzing {tweet_count} tweets..."):
    df = generate_sentiment_data(tweet_count)

# Key metrics
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

positive_count = len(df[df['sentiment'] == 'POSITIVE'])
negative_count = len(df[df['sentiment'] == 'NEGATIVE'])
neutral_count = len(df[df['sentiment'] == 'NEUTRAL'])
avg_confidence = df['confidence'].mean()

with col1:
    st.metric(
        label="✅ Positive Tweets",
        value=positive_count,
        delta=f"{positive_count/len(df)*100:.1f}%"
    )

with col2:
    st.metric(
        label="❌ Negative Tweets", 
        value=negative_count,
        delta=f"{negative_count/len(df)*100:.1f}%"
    )

with col3:
    st.metric(
        label="⚪ Neutral Tweets",
        value=neutral_count,
        delta=f"{neutral_count/len(df)*100:.1f}%"
    )

with col4:
    st.metric(
        label="🎯 Avg Confidence",
        value=f"{avg_confidence:.3f}",
        delta="High Quality"
    )

# Charts section
st.subheader("📈 Sentiment Analysis")

# Two column layout for charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create pie chart data
    chart_data = pd.DataFrame({
        'Sentiment': sentiment_counts.index,
        'Count': sentiment_counts.values
    })
    
    st.bar_chart(chart_data.set_index('Sentiment'))

with chart_col2:
    st.subheader("Confidence Levels")
    
    # Confidence histogram
    confidence_data = pd.DataFrame({
        'Confidence': df['confidence']
    })
    
    st.line_chart(df.groupby(df['confidence'].round(2)).size())

# Topic analysis
st.subheader("🏷️ Topic Analysis")

topic_sentiment = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
st.bar_chart(topic_sentiment)

# Recent tweets table
st.subheader("📝 Recent Tweet Analysis")

# Display sample tweets
sample_tweets = df.head(20)[['text', 'sentiment', 'confidence', 'topic', 'likes', 'retweets']]
sample_tweets['confidence'] = sample_tweets['confidence'].round(3)

st.dataframe(
    sample_tweets,
    use_container_width=True,
    hide_index=True
)

# Performance info
st.subheader("⚡ Performance Info")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"📊 **Tweets Processed:** {len(df):,}")

with col2:
    st.info(f"🚀 **Processing Speed:** ~{len(df)/2:.0f} tweets/sec")

with col3:
    st.info(f"💾 **Memory Usage:** Optimized for Cloud")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🐦 <strong>Twitter Sentiment Analysis Dashboard</strong></p>
    <p>Built with Streamlit • Real-time Processing • Cloud Optimized</p>
    <p><em>Demo mode with synthetic data for demonstration purposes</em></p>
</div>
""", unsafe_allow_html=True)

# Success message
st.success("✅ Dashboard loaded successfully! This is a demo version with synthetic data.")
st.info("💡 **Note:** This version runs on Streamlit Cloud with CPU processing. For GPU-accelerated analysis, run locally with the full codebase.")
