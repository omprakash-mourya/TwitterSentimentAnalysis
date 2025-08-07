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

# Custom CSS for dark theme styling
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
        border: 1px solid #404040;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #9E9E9E;
        font-weight: bold;
    }
    .warning-box {
        background-color: #2D1B1B;
        border: 1px solid #5D4037;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #FFAB91;
    }
    /* Dark theme enhancements */
    .stSelectbox > div > div {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stNumberInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #404040;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #404040;
    }
    /* Progress bar styling for dark theme */
    .stProgress > div > div > div {
        background-color: #1DA1F2;
    }
    /* Tab styling for dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #404040;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1DA1F2;
        color: white;
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
if 'chunk_progress' not in st.session_state:
    st.session_state.chunk_progress = 0
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0
if 'processing_start_time' not in st.session_state:
    st.session_state.processing_start_time = None
if 'connection_check' not in st.session_state:
    st.session_state.connection_check = time.time()

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
    if df.empty:
        st.warning("📊 No data available for pie chart")
        return None
        
    if 'sentiment' not in df.columns:
        st.error("❌ Sentiment analysis results not found. Please re-run the analysis.")
        return None
    
    sentiment_counts = df['sentiment'].value_counts()
    
    if sentiment_counts.empty:
        st.warning("📊 No sentiment data to display")
        return None
    
    # Define colors for sentiments (dark theme compatible)
    colors = {
        'POSITIVE': '#4CAF50',
        'NEGATIVE': '#F44336', 
        'NEUTRAL': '#9E9E9E'
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
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textfont_size=14,
        textfont_color='white'
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=14, color='#FAFAFA'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_color='#FAFAFA',
        legend=dict(
            font=dict(color='#FAFAFA')
        )
    )
    
    return fig

def create_sentiment_bar_chart(df):
    """Create a bar chart showing sentiment distribution"""
    if df.empty:
        st.warning("📈 No data available for bar chart")
        return None
        
    if 'sentiment' not in df.columns:
        st.error("❌ Sentiment analysis results not found. Please re-run the analysis.")
        return None
    
    sentiment_counts = df['sentiment'].value_counts()
    
    if sentiment_counts.empty:
        st.warning("📈 No sentiment data to display")
        return None
    
    colors = {
        'POSITIVE': '#4CAF50',
        'NEGATIVE': '#F44336',
        'NEUTRAL': '#9E9E9E'
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
        yaxis_title="Number of Tweets",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        title_font_color='#FAFAFA',
        xaxis=dict(
            color='#FAFAFA',
            gridcolor='#404040'
        ),
        yaxis=dict(
            color='#FAFAFA',
            gridcolor='#404040'
        )
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
    
    # Create word cloud with dark theme
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#0E1117',  # Dark background
        stopwords=STOPWORDS,
        max_words=100,
        relative_scaling=0.5,
        colormap='viridis'
    ).generate(cleaned_text)
    
    # Create matplotlib figure with dark theme
    try:
        plt.style.use('dark_background')
    except:
        pass  # Fallback if style not available
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud{" - " + sentiment_filter if sentiment_filter else ""}', 
                 fontsize=16, pad=20, color='#FAFAFA')
    
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
        
        # Number of tweets - with options for large-scale analysis
        analysis_scale = st.radio(
            "📊 Analysis Scale:",
            options=["Quick Demo (≤200 tweets)", "Medium Scale (≤10K tweets)", "Large Scale (≤1M tweets)"],
            index=0,
            help="Choose analysis scale based on your needs"
        )
        
        if analysis_scale == "Quick Demo (≤200 tweets)":
            num_tweets = st.slider(
                "Number of tweets to analyze:",
                min_value=10,
                max_value=200,
                value=50,
                help="Fast analysis for quick insights"
            )
        elif analysis_scale == "Medium Scale (≤10K tweets)":
            num_tweets = st.number_input(
                "Number of tweets to analyze:",
                min_value=201,
                max_value=10000,
                value=1000,
                step=100,
                help="Medium-scale analysis (~30 seconds to 2 minutes)"
            )
        else:  # Large Scale
            num_tweets = st.number_input(
                "Number of tweets to analyze:",
                min_value=10001,
                max_value=1000000,
                value=50000,
                step=5000,
                help="⚠️ Large-scale analysis may take 10-30 minutes depending on your hardware"
            )
            
            if num_tweets > 100000:
                st.warning(
                    f"🚨 **High Volume Analysis**: You've selected {num_tweets:,} tweets. "
                    f"This will take approximately {(num_tweets/1000)*0.5:.1f}-{(num_tweets/1000)*2:.1f} minutes to complete."
                )
        
        # Show estimated processing time
        if num_tweets > 1000:
            estimated_time = (num_tweets / 1000) * 0.5  # ~0.5 minutes per 1000 tweets
            if estimated_time < 1:
                time_str = f"{estimated_time*60:.0f} seconds"
            else:
                time_str = f"{estimated_time:.1f} minutes"
            st.info(f"⏱️ Estimated processing time: ~{time_str}")
        
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
            
            # Show progress for tweet scraping
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"🔍 Searching for {num_tweets:,} tweets about '{query}'...")
            progress_bar.progress(10)
            
            df = scraper.scrape_tweets(query, max_tweets=num_tweets, days_back=days_back)
            progress_bar.progress(30)
            
            if df.empty:
                st.error(f"No tweets found for '{query}'. Try a different keyword or increase the time range.")
                st.stop()
            
            status_text.text(f"✅ Found {len(df):,} tweets! Now cleaning and analyzing...")
            progress_bar.progress(50)
        
        # Clean and analyze sentiment with progress tracking
        if 'content' in df.columns:
            status_text.text("🧹 Cleaning tweet content...")
            df['cleaned_content'] = df['content'].apply(cleaner.clean_text)
            progress_bar.progress(70)
        
        # Analyze sentiment with chunked processing to prevent timeouts
        if len(df) > 1000:
            status_text.text(f"🤖 Analyzing sentiment for {len(df):,} tweets with GPU acceleration...")
            
            # Initialize progress tracking
            if 'chunk_progress' not in st.session_state:
                st.session_state.chunk_progress = 0
                st.session_state.total_chunks = 0
                st.session_state.processed_chunks = []
            
            # Process in chunks to prevent connection timeout
            chunk_size = 500  # Process 500 tweets at a time
            total_chunks = (len(df) + chunk_size - 1) // chunk_size
            st.session_state.total_chunks = total_chunks
            
            # Create a placeholder for chunk progress
            chunk_status = st.empty()
            
            # Store processed chunks to rebuild dataframe
            processed_chunks = []
            
            for i in range(0, len(df), chunk_size):
                chunk_num = i // chunk_size + 1
                end_idx = min(i + chunk_size, len(df))
                chunk_df = df.iloc[i:end_idx].copy()
                
                # Update progress display
                chunk_status.text(f"Processing chunk {chunk_num}/{total_chunks} (tweets {i+1}-{end_idx})...")
                progress_percentage = 75 + (chunk_num / total_chunks) * 20  # 75-95% range
                progress_bar.progress(int(progress_percentage))
                
                # Process chunk
                chunk_df = analyzer.analyze_dataframe(chunk_df, text_column='content')
                
                # Store processed chunk
                processed_chunks.append(chunk_df)
                
                # Force a small delay to prevent overwhelming the connection
                time.sleep(0.1)
                
                # Update session state
                st.session_state.chunk_progress = chunk_num
            
            # Rebuild the complete dataframe from processed chunks
            df = pd.concat(processed_chunks, ignore_index=True)
            chunk_status.empty()
        else:
            status_text.text("🤖 Analyzing sentiment...")
            df = analyzer.analyze_dataframe(df, text_column='content')
        
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state.tweets_df = df
        st.session_state.last_query = query
        st.session_state.analysis_complete = True
        
        # Debug information for large datasets
        if len(df) > 1000:
            st.write(f"🔍 Debug Info - DataFrame columns: {list(df.columns)}")
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                st.write(f"📊 Sentiment distribution: {dict(sentiment_counts)}")
            else:
                st.error("❌ 'sentiment' column missing from processed data!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show completion message with stats
        total_tweets = len(df)
        processing_time = "estimated"  # In real implementation, track actual time
        st.balloons()
        st.success(f"🎉 Analysis complete! Processed {total_tweets:,} tweets successfully!")
    
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
                        'POSITIVE': '#4CAF50',
                        'NEGATIVE': '#F44336',
                        'NEUTRAL': '#9E9E9E'
                    }
                )
                
                # Update layout for dark theme
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FAFAFA'),
                    title_font_color='#FAFAFA',
                    xaxis=dict(
                        color='#FAFAFA',
                        gridcolor='#404040'
                    ),
                    yaxis=dict(
                        color='#FAFAFA',
                        gridcolor='#404040'
                    ),
                    legend=dict(
                        font=dict(color='#FAFAFA')
                    )
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
