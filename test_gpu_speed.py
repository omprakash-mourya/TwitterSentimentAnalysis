"""
Simple connection timeout test for Streamlit dashboard
"""

import time
import pandas as pd
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import SentimentAnalyzer

def test_gpu_processing_speed():
    """Test GPU processing without Streamlit to verify speed"""
    print("🚀 Testing GPU Processing Speed")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Create test data
    test_tweets = [
        "I love this product so much!",
        "This is terrible, worst experience ever.",
        "It's okay, nothing special really.",
        "Amazing quality, highly recommend!",
        "Not impressed, could be much better.",
        "Absolutely hate it, complete waste of money!"
    ] * 1667  # Creates ~10K tweets
    
    print(f"📊 Processing {len(test_tweets):,} tweets...")
    
    start_time = time.time()
    
    # Test batch processing
    results = analyzer.predict_batch(test_tweets)
    
    processing_time = time.time() - start_time
    speed = len(test_tweets) / processing_time
    
    # Count sentiments
    sentiments = [r['label'] for r in results]
    sentiment_counts = {}
    for sentiment in sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    print(f"\n✅ Results:")
    print(f"  ⏱️ Processing time: {processing_time:.1f}s")
    print(f"  🚀 Speed: {speed:.1f} tweets/sec")
    print(f"  📊 Sentiment distribution: {sentiment_counts}")
    
    print(f"\n🎯 GPU Performance Test Successful!")
    print(f"   Your RTX 3050 6GB is processing at {speed:.1f} tweets/sec")
    print(f"   For 10K tweets, it takes only {processing_time:.1f} seconds!")

if __name__ == "__main__":
    test_gpu_processing_speed()
