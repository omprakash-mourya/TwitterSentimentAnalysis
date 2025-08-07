"""
Streamlit Cloud Compatible Entry Point
GPU-Accelerated Twitter Sentiment Analysis Dashboard
"""

import streamlit as st
import sys
import os
import traceback

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set page config first
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analysis - Cloud",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import the main application with error handling
try:
    # Import main modules with fallback
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    import time
    import random
    
    # Try to import the full app
    try:
        from app.streamlit_app import *
    except ImportError as e:
        st.warning(f"⚠️ Full app import failed: {e}")
        st.info("🔄 Running in simplified cloud mode...")
        
        # Simplified cloud-only version
        st.title("🐦 Twitter Sentiment Analysis - Cloud Mode")
        st.markdown("---")
        
        # Mock data for demonstration
        @st.cache_data
        def generate_demo_data(num_tweets=1000):
            """Generate demo sentiment data"""
            tweets = []
            sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
            topics = ['technology', 'sports', 'politics', 'entertainment', 'business']
            
            for i in range(num_tweets):
                sentiment = random.choice(sentiments)
                confidence = random.uniform(0.6, 0.95)
                topic = random.choice(topics)
                
                tweets.append({
                    'text': f"Demo tweet {i+1} about {topic}",
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 1440))
                })
            
            return pd.DataFrame(tweets)
        
        # Generate demo data
        df = generate_demo_data()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        positive_count = len(df[df['sentiment'] == 'POSITIVE'])
        negative_count = len(df[df['sentiment'] == 'NEGATIVE'])
        neutral_count = len(df[df['sentiment'] == 'NEUTRAL'])
        avg_confidence = df['confidence'].mean()
        
        with col1:
            st.metric("✅ Positive", positive_count, delta=f"{positive_count/len(df)*100:.1f}%")
        
        with col2:
            st.metric("❌ Negative", negative_count, delta=f"{negative_count/len(df)*100:.1f}%")
        
        with col3:
            st.metric("⚪ Neutral", neutral_count, delta=f"{neutral_count/len(df)*100:.1f}%")
        
        with col4:
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}", delta="Demo Mode")
        
        # Sentiment distribution chart
        st.subheader("📊 Sentiment Distribution")
        
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Tweet Sentiment Distribution",
            color_discrete_map={
                'POSITIVE': '#00D4AA',
                'NEGATIVE': '#FF6B6B',
                'NEUTRAL': '#4ECDC4'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        st.subheader("🎯 Confidence Distribution")
        fig2 = px.histogram(
            df, 
            x='confidence', 
            color='sentiment',
            title="Sentiment Confidence Distribution",
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Recent tweets sample
        st.subheader("📝 Recent Tweets (Demo)")
        st.dataframe(
            df.head(10)[['text', 'sentiment', 'confidence']],
            use_container_width=True
        )
        
        st.success("✅ Cloud deployment successful! This is a demo version.")
        st.info("💡 For full GPU-accelerated features, run locally with: `streamlit run app/streamlit_app.py`")

except Exception as e:
    st.error(f"❌ Critical error: {e}")
    st.code(traceback.format_exc())
    st.info("Please check the deployment logs for more details.")
