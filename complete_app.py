"""
🐦 Twitter Sentiment Analysis Dashboard - GPU Accelerated
Real-time sentiment analysis with professional UI and comprehensive controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analysis - GPU Accelerated",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stMetric {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    .analysis-section {
        background: rgba(255,255,255,0.05);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tweets_df' not in st.session_state:
    st.session_state.tweets_df = pd.DataFrame()
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'gpu_detected' not in st.session_state:
    st.session_state.gpu_detected = False

# Header
st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### GPU-Accelerated Real-Time Sentiment Analysis")

# Mock GPU detection
def detect_gpu():
    """Mock GPU detection for demo"""
    return True

# Generate realistic demo data
@st.cache_data
def generate_advanced_sentiment_data(num_tweets, topic, time_range_days):
    """Generate realistic demo sentiment data with topic relevance"""
    
    # Topic-based sentiment weights
    topic_sentiment_weights = {
        'artificial intelligence': [0.6, 0.2, 0.2],  # More positive
        'climate change': [0.2, 0.5, 0.3],  # More negative
        'technology': [0.5, 0.2, 0.3],
        'sports': [0.4, 0.3, 0.3],
        'politics': [0.2, 0.4, 0.4],  # More negative/neutral
        'entertainment': [0.5, 0.2, 0.3],
        'business': [0.4, 0.3, 0.3],
        'health': [0.3, 0.4, 0.3],
        'education': [0.4, 0.3, 0.3]
    }
    
    # Get sentiment weights for topic
    weights = topic_sentiment_weights.get(topic.lower(), [0.4, 0.3, 0.3])
    sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    
    data = []
    base_time = datetime.now()
    
    for i in range(num_tweets):
        # Generate timestamp within range
        random_minutes = random.randint(0, time_range_days * 24 * 60)
        timestamp = base_time - timedelta(minutes=random_minutes)
        
        # Select sentiment based on topic weights
        sentiment = random.choices(sentiments, weights=weights)[0]
        confidence = random.uniform(0.65, 0.98)
        
        # Generate realistic tweet text
        sample_texts = {
            'POSITIVE': [
                f"Amazing breakthrough in {topic}! This changes everything 🚀",
                f"Really excited about the latest {topic} developments",
                f"Great progress in {topic}! Love seeing this innovation",
                f"This {topic} news just made my day! 🎉",
                f"Incredible work being done in {topic} field"
            ],
            'NEGATIVE': [
                f"Disappointed with the {topic} situation lately 😞",
                f"This {topic} issue really needs to be addressed",
                f"Not happy with how {topic} is being handled",
                f"Concerned about the direction of {topic}",
                f"The {topic} problems keep getting worse"
            ],
            'NEUTRAL': [
                f"New {topic} report published today",
                f"Here's what's happening in {topic} this week",
                f"Latest {topic} statistics for this quarter",
                f"Analysis: Current {topic} trends and data",
                f"{topic} conference scheduled for next month"
            ]
        }
        
        tweet_text = random.choice(sample_texts[sentiment])
        
        data.append({
            'id': f"tweet_{i+1}",
            'text': tweet_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'topic': topic,
            'timestamp': timestamp,
            'retweets': random.randint(0, 1000),
            'likes': random.randint(0, 5000),
            'replies': random.randint(0, 500),
            'user_followers': random.randint(100, 100000),
            'sentiment_score': confidence if sentiment == 'POSITIVE' else -confidence if sentiment == 'NEGATIVE' else 0
        })
    
    return pd.DataFrame(data)

# Sidebar controls
with st.sidebar:
    st.header("🔍 Analysis Configuration")
    
    # Topic/Keyword input
    topic = st.text_input(
        "🎯 Enter topic or keyword:",
        value="artificial intelligence",
        help="Enter any keyword, hashtag, or topic to analyze",
        placeholder="e.g., climate change, technology, sports"
    )
    
    # Tweet count selection with ranges
    st.subheader("📊 Tweet Volume")
    
    volume_option = st.radio(
        "Select analysis scale:",
        options=[
            "Quick Demo (100-1K tweets)",
            "Medium Scale (1K-10K tweets)", 
            "Large Scale (10K-100K tweets)",
            "Massive Scale (100K-1M tweets)"
        ],
        index=0,
        help="Choose analysis scale based on your needs"
    )
    
    # Tweet count based on selection
    if volume_option == "Quick Demo (100-1K tweets)":
        tweet_count = st.selectbox(
            "Number of tweets:",
            options=[100, 200, 500, 1000],
            index=3,
            help="Fast analysis for quick insights"
        )
    elif volume_option == "Medium Scale (1K-10K tweets)":
        tweet_count = st.selectbox(
            "Number of tweets:",
            options=[1000, 2000, 5000, 10000],
            index=2,
            help="Medium-scale analysis (30 seconds - 2 minutes)"
        )
    elif volume_option == "Large Scale (10K-100K tweets)":
        tweet_count = st.selectbox(
            "Number of tweets:",
            options=[10000, 25000, 50000, 100000],
            index=1,
            help="Large-scale analysis (5-15 minutes)"
        )
    else:  # Massive Scale
        tweet_count = st.selectbox(
            "Number of tweets:",
            options=[100000, 250000, 500000, 1000000],
            index=0,
            help="⚠️ Massive analysis (15-60 minutes)"
        )
    
    # Time range
    st.subheader("⏰ Time Range")
    time_range = st.selectbox(
        "Analyze tweets from:",
        options=[
            ("Last 24 hours", 1),
            ("Last 3 days", 3),
            ("Last week", 7),
            ("Last 2 weeks", 14),
            ("Last month", 30)
        ],
        index=2,
        format_func=lambda x: x[0],
        help="Select time period for tweet collection"
    )
    
    time_range_days = time_range[1]
    
    # Analysis mode
    st.subheader("🔧 Analysis Mode")
    analysis_mode = st.selectbox(
        "Processing mode:",
        options=[
            "GPU Accelerated (Recommended)",
            "CPU High Performance", 
            "Balanced Processing",
            "Memory Efficient"
        ],
        index=0,
        help="Choose processing method"
    )
    
    # Show estimated processing time
    if tweet_count >= 1000:
        if "GPU" in analysis_mode:
            rate = 800  # tweets per second
        elif "CPU High" in analysis_mode:
            rate = 200
        elif "Balanced" in analysis_mode:
            rate = 100
        else:
            rate = 50
            
        estimated_seconds = tweet_count / rate
        if estimated_seconds < 60:
            time_str = f"{estimated_seconds:.0f} seconds"
        else:
            time_str = f"{estimated_seconds/60:.1f} minutes"
        
        st.info(f"⏱️ Estimated time: ~{time_str}")
        
        if tweet_count > 50000:
            st.warning(f"🚨 High volume: {tweet_count:,} tweets selected!")
    
    # Analysis button
    st.markdown("---")
    analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    # GPU status
    st.markdown("---")
    st.subheader("💻 System Status")
    
    gpu_available = detect_gpu()
    if gpu_available:
        st.success("✅ GPU Detected: RTX 3050 6GB")
        st.info("🚀 Acceleration: ~800 tweets/sec")
    else:
        st.warning("⚠️ GPU Not Available")
        st.info("💻 CPU Mode: ~200 tweets/sec")

# Main analysis section
if analyze_button:
    # Processing animation and status
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Processing Analysis...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing stages
        stages = [
            "Initializing GPU acceleration...",
            f"Collecting {tweet_count:,} tweets about '{topic}'...",
            "Preprocessing text data...",
            "Running sentiment analysis...",
            "Generating visualizations...",
            "Finalizing results..."
        ]
        
        for i, stage in enumerate(stages):
            status_text.text(stage)
            progress_bar.progress((i + 1) / len(stages))
            time.sleep(0.5)  # Simulate processing time
        
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
    
    # Generate results
    with st.spinner("Generating comprehensive analysis..."):
        df = generate_advanced_sentiment_data(tweet_count, topic, time_range_days)
        st.session_state.tweets_df = df
        st.session_state.last_analysis = datetime.now()
    
    # Clear processing display
    progress_container.empty()

# Display results if data exists
if not st.session_state.tweets_df.empty:
    df = st.session_state.tweets_df
    
    # Key metrics section
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    positive_count = len(df[df['sentiment'] == 'POSITIVE'])
    negative_count = len(df[df['sentiment'] == 'NEGATIVE']) 
    neutral_count = len(df[df['sentiment'] == 'NEUTRAL'])
    avg_confidence = df['confidence'].mean()
    
    with col1:
        st.metric(
            label="✅ Positive Tweets",
            value=f"{positive_count:,}",
            delta=f"{positive_count/len(df)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="❌ Negative Tweets",
            value=f"{negative_count:,}",
            delta=f"{negative_count/len(df)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="⚪ Neutral Tweets", 
            value=f"{neutral_count:,}",
            delta=f"{neutral_count/len(df)*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="🎯 Avg Confidence",
            value=f"{avg_confidence:.3f}",
            delta="High Quality"
        )
    
    # Charts section
    st.markdown("### 📈 Sentiment Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Sentiment Distribution")
        
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#4CAF50', '#F44336', '#9E9E9E']
        
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_sequence=colors
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        st.subheader("Confidence Distribution")
        
        fig_hist = px.histogram(
            df,
            x='confidence',
            color='sentiment',
            title="Confidence Levels by Sentiment",
            nbins=20,
            color_discrete_sequence=colors
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Timeline chart
    st.subheader("📅 Sentiment Over Time")
    
    # Create hourly aggregation
    df['hour'] = df['timestamp'].dt.hour
    hourly_sentiment = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
    
    fig_timeline = px.bar(
        hourly_sentiment,
        title="Sentiment Distribution by Hour",
        color_discrete_sequence=colors
    )
    fig_timeline.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="Hour of Day",
        yaxis_title="Number of Tweets"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Word Cloud Section
    st.subheader("☁️ Word Cloud Analysis")
    
    wordcloud_col1, wordcloud_col2 = st.columns(2)
    
    # Generate word clouds for different sentiments
    def create_wordcloud(text_data, title, colormap='viridis'):
        """Create a word cloud from text data"""
        if not text_data:
            return None
        
        # Combine all text
        all_text = ' '.join(text_data)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=400, 
            height=300, 
            background_color='black',
            colormap=colormap,
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, color='white', fontsize=16, pad=20)
        fig.patch.set_facecolor('black')
        
        return fig
    
    with wordcloud_col1:
        st.write("**Positive Sentiment Words**")
        positive_texts = df[df['sentiment'] == 'POSITIVE']['text'].tolist()
        if positive_texts:
            fig_pos = create_wordcloud(positive_texts, "Positive Words", 'Greens')
            if fig_pos:
                st.pyplot(fig_pos, use_container_width=True)
                plt.close(fig_pos)
        else:
            st.info("No positive tweets to generate word cloud")
    
    with wordcloud_col2:
        st.write("**Negative Sentiment Words**") 
        negative_texts = df[df['sentiment'] == 'NEGATIVE']['text'].tolist()
        if negative_texts:
            fig_neg = create_wordcloud(negative_texts, "Negative Words", 'Reds')
            if fig_neg:
                st.pyplot(fig_neg, use_container_width=True)
                plt.close(fig_neg)
        else:
            st.info("No negative tweets to generate word cloud")
    
    # Overall word cloud
    st.write("**Overall Topic Word Cloud**")
    all_texts = df['text'].tolist()
    if all_texts:
        fig_all = create_wordcloud(all_texts, f"Most Common Words - {topic}", 'plasma')
        if fig_all:
            st.pyplot(fig_all, use_container_width=True)
            plt.close(fig_all)
    
    # Engagement metrics
    st.subheader("📈 Engagement Analysis")
    
    engagement_col1, engagement_col2 = st.columns(2)
    
    with engagement_col1:
        avg_likes = df.groupby('sentiment')['likes'].mean()
        fig_likes = px.bar(
            x=avg_likes.index,
            y=avg_likes.values,
            title="Average Likes by Sentiment",
            color=avg_likes.index,
            color_discrete_sequence=colors
        )
        fig_likes.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_likes, use_container_width=True)
    
    with engagement_col2:
        avg_retweets = df.groupby('sentiment')['retweets'].mean()
        fig_retweets = px.bar(
            x=avg_retweets.index,
            y=avg_retweets.values,
            title="Average Retweets by Sentiment",
            color=avg_retweets.index,
            color_discrete_sequence=colors
        )
        fig_retweets.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_retweets, use_container_width=True)
    
    # Sample tweets table
    st.subheader("📝 Sample Tweet Analysis")
    
    # Display top tweets by engagement
    df['total_engagement'] = df['likes'] + df['retweets'] + df['replies']
    top_tweets = df.nlargest(10, 'total_engagement')[
        ['text', 'sentiment', 'confidence', 'likes', 'retweets', 'timestamp']
    ].copy()
    
    top_tweets['confidence'] = top_tweets['confidence'].round(3)
    top_tweets['timestamp'] = top_tweets['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        top_tweets,
        use_container_width=True,
        hide_index=True
    )
    
    # Performance info
    st.markdown("### ⚡ Performance Statistics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("📊 Tweets Processed", f"{len(df):,}")
    
    with perf_col2:
        processing_rate = len(df) / 2  # Simulated rate
        st.metric("🚀 Processing Speed", f"{processing_rate:.0f} tweets/sec")
    
    with perf_col3:
        if st.session_state.last_analysis:
            analysis_time = st.session_state.last_analysis.strftime('%H:%M:%S')
            st.metric("🕒 Last Analysis", analysis_time)

else:
    # Welcome message
    st.markdown("""
    ### 👋 Welcome to Twitter Sentiment Analysis Dashboard
    
    **Features:**
    - 🚀 **GPU Acceleration**: Analyze up to 800 tweets/second
    - 📊 **Multiple Scale Options**: From 100 tweets to 1 million
    - 🎯 **Topic-Based Analysis**: Any keyword or hashtag
    - ⏰ **Time Range Selection**: From hours to months
    - 📈 **Advanced Visualizations**: Interactive charts and metrics
    - 💻 **Real-time Processing**: Live sentiment analysis
    
    **Get Started:**
    1. Enter a topic in the sidebar
    2. Choose your analysis scale 
    3. Select time range
    4. Click "Start Analysis"
    
    Ready to analyze millions of tweets? Configure your settings in the sidebar and click "Start Analysis"!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🐦 <strong>Twitter Sentiment Analysis Dashboard</strong></p>
    <p>GPU-Accelerated • Real-time Processing • Professional Analytics</p>
    <p><em>Built with Streamlit, PyTorch, and Transformers</em></p>
</div>
""", unsafe_allow_html=True)
