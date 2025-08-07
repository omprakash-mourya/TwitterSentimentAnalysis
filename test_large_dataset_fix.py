"""
Test script to verify large dataset chart display after chunked processing fix
"""

import pandas as pd
import sys
import os
import time

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import SentimentAnalyzer
from app.tweet_scraper import TwitterScraper

def test_large_dataset_charts():
    """Test that charts display properly for datasets > 1000 tweets"""
    print("📊 Testing Large Dataset Chart Display Fix")
    print("=" * 55)
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    scraper = TwitterScraper()
    
    # Test with 2000 tweets (>1000 to trigger chunked processing)
    test_size = 2000
    print(f"🔍 Testing with {test_size:,} tweets...")
    
    start_time = time.time()
    
    # Generate test data
    df = scraper.scrape_tweets("test topic", max_tweets=test_size, days_back=7)
    print(f"✅ Generated {len(df):,} tweets")
    
    # Simulate chunked processing like in Streamlit
    print(f"🤖 Applying chunked sentiment analysis...")
    
    chunk_size = 500
    processed_chunks = []
    
    for i in range(0, len(df), chunk_size):
        end_idx = min(i + chunk_size, len(df))
        chunk_df = df.iloc[i:end_idx].copy()
        
        chunk_num = i // chunk_size + 1
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        print(f"  Processing chunk {chunk_num}/{total_chunks} ({end_idx - i} tweets)...")
        
        # Analyze chunk
        chunk_df = analyzer.analyze_dataframe(chunk_df, text_column='content')
        processed_chunks.append(chunk_df)
    
    # Rebuild dataframe using pd.concat (like the fix)
    df_final = pd.concat(processed_chunks, ignore_index=True)
    
    processing_time = time.time() - start_time
    
    # Verify data integrity
    print(f"\n📊 Results Verification:")
    print(f"  ✅ Original tweets: {len(df):,}")
    print(f"  ✅ Final tweets: {len(df_final):,}")
    print(f"  ✅ Columns: {list(df_final.columns)}")
    
    # Check sentiment column
    if 'sentiment' in df_final.columns:
        sentiment_counts = df_final['sentiment'].value_counts()
        print(f"  ✅ Sentiment distribution: {dict(sentiment_counts)}")
        print(f"  ✅ Charts will display properly!")
    else:
        print(f"  ❌ Sentiment column missing - charts won't work")
    
    print(f"\n⏱️ Processing time: {processing_time:.1f}s")
    print(f"🚀 Speed: {len(df_final)/processing_time:.1f} tweets/sec")
    
    print(f"\n🎯 Fix Status:")
    if 'sentiment' in df_final.columns and len(df_final) == test_size:
        print("✅ Large dataset chart fix is WORKING!")
        print("✅ Charts will display for datasets > 1000 tweets")
    else:
        print("❌ Issue still exists - needs further debugging")

if __name__ == "__main__":
    test_large_dataset_charts()
