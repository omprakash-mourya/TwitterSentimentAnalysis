"""
Demo script to test the sentiment analysis components.
This shows how to use the modules independently.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.sentiment_model import analyzer
from utils.text_cleaning import cleaner
import pandas as pd

def demo_sentiment_analysis():
    """Demo the sentiment analysis functionality"""
    print("🤖 Testing Sentiment Analysis Model...")
    
    # Test texts
    test_texts = [
        "I love this new product! It's amazing!",
        "This is terrible, worst experience ever.",
        "It's okay, nothing special.",
        "Absolutely fantastic! Highly recommend!",
        "Not bad, could be better though."
    ]
    
    print("\n📝 Analyzing sample texts:")
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Confidence: {result['score']:.2%})")
        print("-" * 50)

def demo_text_cleaning():
    """Demo the text cleaning functionality"""
    print("\n🧹 Testing Text Cleaning...")
    
    dirty_text = "@user Check out this amazing product! https://example.com #awesome 🔥🔥🔥"
    
    print(f"Original: {dirty_text}")
    print(f"Cleaned: {cleaner.clean_text(dirty_text)}")
    print(f"For Sentiment: {cleaner.preprocess_for_sentiment(dirty_text)}")
    print(f"For Word Cloud: {cleaner.preprocess_for_wordcloud(dirty_text)}")

def demo_dataframe_analysis():
    """Demo analyzing a dataframe"""
    print("\n📊 Testing DataFrame Analysis...")
    
    # Create sample data
    sample_data = {
        'content': [
            "I love this new technology! It's amazing!",
            "This is terrible, worst experience ever.",
            "It's okay, nothing special.",
            "Absolutely fantastic! Highly recommend!",
            "Not bad, could be better though."
        ],
        'user': ['user1', 'user2', 'user3', 'user4', 'user5'],
        'like_count': [10, 5, 3, 25, 8],
        'retweet_count': [2, 1, 0, 8, 3]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Analyze sentiment
    df_analyzed = analyzer.analyze_dataframe(df)
    print("\nWith Sentiment Analysis:")
    print(df_analyzed[['content', 'sentiment', 'sentiment_score']])
    
    # Get summary
    summary = analyzer.get_sentiment_summary(df_analyzed)
    print("\nSentiment Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("🐦 Real-Time Twitter Sentiment Analysis - Demo")
    print("=" * 60)
    
    try:
        demo_text_cleaning()
        demo_sentiment_analysis()
        demo_dataframe_analysis()
        
        print("\n✅ All tests completed successfully!")
        print("\n🚀 Ready to run the full application:")
        print("   streamlit run app/streamlit_app.py")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check your installation and try again.")
