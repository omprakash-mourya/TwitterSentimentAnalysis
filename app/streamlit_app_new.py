"""
Real-Time Twitter Sentiment Analysis Dashboard
Combines the best features from all three projects with modern UI and powerful analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
from datetime import datetime, timedelta
import time
import sys
import os

# Add project directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our custom modules
try:
    from app.tweet_scraper import scraper
    from models.sentiment_model import analyzer
    from utils.text_cleaning import cleaner
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🐦 Real-Time Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1DA1F2;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #1DA1F2 0%, #14171A 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tweets_df' not in st.session_state:
    st.session_state.tweets_df = pd.DataFrame()
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def load_sample_data():
    """Load sample data if no real data is available"""
    sample_data = {
        'content': [
            "I love this new technology! It's amazing!",
            "This is terrible, worst experience ever.",
            "It's okay, nothing special.",
            "Absolutely fantastic! Highly recommend!",
            "Not bad, could be better though.",
            "Hate it so much, complete waste of time!",
            "Pretty good overall, satisfied with it.",
            "Disappointing results, expected much more.",
            "Excellent quality and great service!",
            "Average product, meets expectations."
        ],
        'user': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10'],
        'date': [datetime.now() - timedelta(hours=i) for i in range(10)],
        'like_count': [10, 5, 3, 25, 8, 2, 12, 4, 30, 7],
        'retweet_count': [2, 1, 0, 8, 3, 0, 5, 1, 12, 2]
    }
    return pd.DataFrame(sample_data)

def create_sentiment_pie_chart(df):
    """Create an interactive pie chart for sentiment distribution"""
    if 'sentiment' not in df.columns:
        return None
    
    sentiment_counts = df['sentiment'].value_counts()
    
    # Define colors for sentiments
    colors = {
        'POSITIVE': '#28a745',
        'NEGATIVE': '#dc3545', 
        'NEUTRAL': '#6c757d'
    }
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="📊 Sentiment Distribution",
        color_discrete_map=colors,
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=14)
    )
    
    return fig

def create_sentiment_bar_chart(df):
    """Create a bar chart showing sentiment distribution"""
    if 'sentiment' not in df.columns:
        return None
    
    sentiment_counts = df['sentiment'].value_counts()
    
    colors = {
        'POSITIVE': '#28a745',
        'NEGATIVE': '#dc3545',
        'NEUTRAL': '#6c757d'
    }
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="📈 Sentiment Analysis Results",
        labels={'x': 'Sentiment', 'y': 'Number of Tweets'},
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Sentiment",
        yaxis_title="Number of Tweets"
    )
    
    return fig

def create_wordcloud(df, sentiment_filter=None):
    """Create a word cloud from tweet content"""
    if df.empty or 'content' not in df.columns:
        return None
    
    # Filter by sentiment if specified
    if sentiment_filter and 'sentiment' in df.columns:
        df_filtered = df[df['sentiment'] == sentiment_filter]
    else:
        df_filtered = df
    
    if df_filtered.empty:
        return None
    
    # Combine all tweet content
    text_data = df_filtered['content'].fillna('').astype(str)
    all_text = ' '.join(text_data)
    
    # Clean text for word cloud
    cleaned_text = cleaner.preprocess_for_wordcloud(all_text)
    
    if not cleaned_text:
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=100,
        relative_scaling=0.5,
        colormap='viridis'
    ).generate(cleaned_text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud{" - " + sentiment_filter if sentiment_filter else ""}', 
                 fontsize=16, pad=20)
    
    return fig

def display_top_tweets(df, sentiment_type, limit=5):
    """Display top tweets of a specific sentiment"""
    if df.empty or 'sentiment' not in df.columns:
        return
    
    filtered_df = df[df['sentiment'] == sentiment_type]
    
    if filtered_df.empty:
        st.write(f"No {sentiment_type.lower()} tweets found.")
        return
    
    # Sort by engagement (likes + retweets) if available
    if 'like_count' in filtered_df.columns and 'retweet_count' in filtered_df.columns:
        filtered_df['engagement'] = filtered_df['like_count'].fillna(0) + filtered_df['retweet_count'].fillna(0)
        top_tweets = filtered_df.nlargest(limit, 'engagement')
    else:
        top_tweets = filtered_df.head(limit)
    
    for idx, tweet in top_tweets.iterrows():
        with st.expander(f"Tweet from @{tweet.get('user', 'unknown')}"):
            st.write(tweet['content'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("❤️ Likes", tweet.get('like_count', 'N/A'))
            with col2:
                st.metric("🔄 Retweets", tweet.get('retweet_count', 'N/A'))
            with col3:
                confidence = tweet.get('sentiment_score', 0)
                st.metric("🎯 Confidence", f"{confidence:.2%}" if confidence else 'N/A')

def main():
    # Header
    st.markdown('<h1 class="main-header">🐦 Real-Time Twitter Sentiment Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Search Configuration")
        
        # Query input
        query = st.text_input(
            "Enter search topic/keyword:",
            value="artificial intelligence",
            help="Enter any keyword, hashtag, or topic you want to analyze"
        )
        
        # Number of tweets
        num_tweets = st.slider(
            "Number of tweets to analyze:",
            min_value=10,
            max_value=200,
            value=50,
            help="More tweets = better analysis but slower processing"
        )
        
        # Time range
        days_back = st.selectbox(
            "Search tweets from:",
            options=[1, 3, 7, 14],
            index=2,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze Tweets", type="primary")
        
        # Sample data option
        use_sample = st.checkbox("Use sample data for demo", value=False)
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        
        if not st.session_state.tweets_df.empty:
            total_tweets = len(st.session_state.tweets_df)
            st.metric("Total Tweets", total_tweets)
            
            if 'sentiment' in st.session_state.tweets_df.columns:
                sentiment_counts = st.session_state.tweets_df['sentiment'].value_counts()
                
                for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / total_tweets) * 100 if total_tweets > 0 else 0
                    
                    color = {
                        'POSITIVE': '🟢',
                        'NEGATIVE': '🔴',
                        'NEUTRAL': '🟡'
                    }[sentiment]
                    
                    st.metric(
                        f"{color} {sentiment.title()}",
                        f"{count} ({percentage:.1f}%)"
                    )
    
    # Main content area
    if analyze_button or use_sample:
        if use_sample:
            st.info("📋 Using sample data for demonstration")
            df = load_sample_data()
        else:
            if not query.strip():
                st.error("Please enter a search query!")
                st.stop()
            
            # Show progress
            with st.spinner(f"🔍 Searching for tweets about '{query}'..."):
                df = scraper.scrape_tweets(query, max_tweets=num_tweets, days_back=days_back)
            
            if df.empty:
                st.error(f"No tweets found for '{query}'. Try a different keyword or increase the time range.")
                st.stop()
            
            st.success(f"✅ Found {len(df)} tweets!")
        
        # Clean and analyze sentiment
        with st.spinner("🤖 Analyzing sentiment..."):
            # Clean tweets
            if 'content' in df.columns:
                df['cleaned_content'] = df['content'].apply(cleaner.clean_text)
            
            # Analyze sentiment
            df = analyzer.analyze_dataframe(df)
        
        # Store in session state
        st.session_state.tweets_df = df
        st.session_state.last_query = query
        st.session_state.analysis_complete = True
        
        st.success("🎉 Analysis complete!")
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and not st.session_state.tweets_df.empty:
        df = st.session_state.tweets_df
        
        # Alert for negative sentiment spike
        if 'sentiment' in df.columns:
            negative_percentage = (df['sentiment'] == 'NEGATIVE').mean() * 100
            if negative_percentage > 60:
                st.markdown(
                    f'<div class="warning-box">⚠️ <strong>High Negative Sentiment Alert!</strong><br>'
                    f'<strong>{negative_percentage:.1f}%</strong> of tweets are negative. '
                    f'This topic may be controversial or problematic.</div>',
                    unsafe_allow_html=True
                )
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "📈 Charts", "☁️ Word Cloud", "🐦 Top Tweets"
        ])
        
        with tab1:
            st.header("📊 Analysis Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📱 Total Tweets", len(df))
            
            with col2:
                avg_engagement = (df.get('like_count', pd.Series([0])).fillna(0) + 
                                df.get('retweet_count', pd.Series([0])).fillna(0)).mean()
                st.metric("📊 Avg Engagement", f"{avg_engagement:.1f}")
            
            with col3:
                if 'sentiment_score' in df.columns:
                    avg_confidence = df['sentiment_score'].mean()
                    st.metric("🎯 Avg Confidence", f"{avg_confidence:.2%}")
                else:
                    st.metric("🎯 Avg Confidence", "N/A")
            
            with col4:
                unique_users = df['user'].nunique() if 'user' in df.columns else 0
                st.metric("👥 Unique Users", unique_users)
            
            # Sentiment distribution
            st.subheader("🎭 Sentiment Distribution")
            if 'sentiment' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    pie_fig = create_sentiment_pie_chart(df)
                    if pie_fig:
                        st.plotly_chart(pie_fig, use_container_width=True)
                
                with col2:
                    # Bar chart
                    bar_fig = create_sentiment_bar_chart(df)
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)
        
        with tab2:
            st.header("📈 Detailed Charts")
            
            # Sentiment vs Engagement scatter plot
            if all(col in df.columns for col in ['sentiment_score', 'like_count', 'retweet_count']):
                df['total_engagement'] = df['like_count'].fillna(0) + df['retweet_count'].fillna(0)
                
                fig = px.scatter(
                    df,
                    x='sentiment_score',
                    y='total_engagement',
                    color='sentiment',
                    title="🎯 Sentiment Confidence vs Engagement",
                    labels={
                        'sentiment_score': 'Sentiment Confidence',
                        'total_engagement': 'Total Engagement (Likes + Retweets)'
                    },
                    color_discrete_map={
                        'POSITIVE': '#28a745',
                        'NEGATIVE': '#dc3545',
                        'NEUTRAL': '#6c757d'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("☁️ Word Cloud Analysis")
            
            # Sentiment filter for word cloud
            sentiment_filter = st.selectbox(
                "Filter by sentiment:",
                options=[None, 'POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                format_func=lambda x: "All Sentiments" if x is None else x.title()
            )
            
            # Generate word cloud
            wordcloud_fig = create_wordcloud(df, sentiment_filter)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.warning("Unable to generate word cloud. Not enough text data.")
        
        with tab4:
            st.header("🐦 Top Tweets by Sentiment")
            
            # Create columns for different sentiments
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🟢 Most Positive")
                display_top_tweets(df, 'POSITIVE')
            
            with col2:
                st.subheader("🔴 Most Negative")
                display_top_tweets(df, 'NEGATIVE')
            
            with col3:
                st.subheader("🟡 Most Neutral")
                display_top_tweets(df, 'NEUTRAL')
            
            # Raw data table
            st.subheader("📋 Raw Data")
            if st.checkbox("Show raw tweet data"):
                # Display selected columns
                display_columns = ['content', 'sentiment', 'sentiment_score', 'user', 'date']
                available_columns = [col for col in display_columns if col in df.columns]
                
                st.dataframe(
                    df[available_columns].head(20),
                    use_container_width=True
                )
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name=f"tweets_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            🚀 Built with Streamlit • 🤖 Powered by HuggingFace Transformers • 🐦 Data from Twitter
            <br>
            <em>Real-time sentiment analysis without API limitations</em>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
