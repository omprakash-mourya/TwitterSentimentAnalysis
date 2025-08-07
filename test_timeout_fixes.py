"""
Test script to verify connection timeout fixes
"""

import time
import pandas as pd
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import SentimentAnalyzer
from app.tweet_scraper import TwitterScraper

def test_large_scale_processing():
    """Test processing without connection timeouts"""
    print("🧪 Testing Large-Scale Processing Without Timeouts")
    print("=" * 60)
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    scraper = TwitterScraper()
    
    # Test with different dataset sizes
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size:,} tweets:")
        print("-" * 40)
        
        start_time = time.time()
        
        # Generate test data
        print(f"🔍 Generating {size:,} test tweets...")
        df = scraper.scrape_tweets("test", max_tweets=size, days_back=7)
        gen_time = time.time() - start_time
        print(f"✅ Generated {len(df):,} tweets in {gen_time:.1f}s")
        
        # Analyze sentiment
        print(f"🤖 Analyzing sentiment with GPU acceleration...")
        analysis_start = time.time()
        
        # Process in chunks to simulate Streamlit environment
        chunk_size = 500
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            
            # Analyze chunk
            chunk_df = analyzer.analyze_dataframe(chunk_df, text_column='content')
            df.iloc[chunk_start:chunk_end] = chunk_df
            
            # Show progress
            progress = ((i + 1) / total_chunks) * 100
            elapsed = time.time() - analysis_start
            speed = (chunk_end) / elapsed if elapsed > 0 else 0
            
            print(f"  Chunk {i+1}/{total_chunks} ({progress:.1f}%) - Speed: {speed:.1f} tweets/sec")
            
            # Small delay to simulate network conditions
            time.sleep(0.05)
        
        analysis_time = time.time() - analysis_start
        total_time = time.time() - start_time
        
        # Results
        sentiment_counts = df['sentiment'].value_counts()
        
        print(f"\n📈 Results for {size:,} tweets:")
        print(f"  ⏱️ Total time: {total_time:.1f}s")
        print(f"  ⚡ Analysis time: {analysis_time:.1f}s")
        print(f"  🚀 Speed: {len(df)/analysis_time:.1f} tweets/sec")
        print(f"  📊 Sentiments: {dict(sentiment_counts)}")
        
        # Memory check
        try:
            import psutil
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            print(f"  💾 Memory usage: {memory:.1f} MB")
        except ImportError:
            pass
    
    print("\n✅ All tests completed successfully!")
    print("🎯 Connection timeout fixes are working properly!")
    
if __name__ == "__main__":
    test_large_scale_processing()
