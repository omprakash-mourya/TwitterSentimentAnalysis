"""
Aggressive GPU Performance Test for Twitter Sentiment Analysis
Tests the maximum GPU utilization and performance improvements.
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import (
    SentimentAnalyzer, 
    get_gpu_memory_info, 
    optimize_gpu_for_aggressive_processing,
    force_gpu_memory_cleanup
)
import torch
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_aggressive_gpu_performance():
    """Test aggressive GPU optimization performance."""
    
    print("🚀 AGGRESSIVE GPU PERFORMANCE TEST")
    print("=" * 60)
    
    # Initialize GPU optimization
    gpu_available = optimize_gpu_for_aggressive_processing()
    
    if gpu_available:
        gpu_info = get_gpu_memory_info()
        print(f"🎯 GPU: {gpu_info['device_name']}")
        print(f"💾 Total Memory: {gpu_info['total_memory_gb']}GB")
        print(f"🔥 Target: 90-100% GPU utilization")
        print("=" * 60)
    else:
        print("💻 CPU-only testing mode")
        print("=" * 60)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test datasets of increasing size
    test_sizes = [100, 500, 1000, 2000, 5000]
    if gpu_available:
        test_sizes.extend([10000, 25000])  # Add larger tests for GPU
    
    results = []
    
    for size in test_sizes:
        print(f"\n🧪 Testing {size:,} tweets...")
        
        # Generate test data
        test_texts = [
            f"This is test tweet number {i} with some sentiment content for analysis. "
            f"It should be processed efficiently with GPU acceleration and aggressive memory usage. "
            f"The sentiment could be positive, negative, or neutral depending on the content."
            for i in range(size)
        ]
        
        # Show GPU memory before processing
        if gpu_available:
            gpu_before = get_gpu_memory_info()
            print(f"   📊 GPU Memory Before: {gpu_before['utilization_percentage']}%")
        
        # Measure processing time
        start_time = time.time()
        predictions = analyzer.predict_batch(test_texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        speed = size / processing_time
        
        # Show GPU memory after processing
        if gpu_available:
            gpu_after = get_gpu_memory_info()
            print(f"   🎯 GPU Memory Peak: {gpu_after['utilization_percentage']}%")
            
            # Clean up for next test
            force_gpu_memory_cleanup()
            gpu_cleaned = get_gpu_memory_info()
            print(f"   🧹 GPU Memory After Cleanup: {gpu_cleaned['utilization_percentage']}%")
        
        # Results
        sentiment_counts = {}
        for pred in predictions:
            label = pred['label']
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        
        result = {
            'size': size,
            'time': processing_time,
            'speed': speed,
            'positive': sentiment_counts.get('POSITIVE', 0),
            'negative': sentiment_counts.get('NEGATIVE', 0),
            'neutral': sentiment_counts.get('NEUTRAL', 0),
            'gpu_utilization': gpu_after['utilization_percentage'] if gpu_available else 0
        }
        results.append(result)
        
        print(f"   ⚡ Speed: {speed:.1f} tweets/sec")
        print(f"   📊 Results: {sentiment_counts}")
        print(f"   ⏱️  Time: {processing_time:.2f}s")
        
        # Pause between tests
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 AGGRESSIVE GPU PERFORMANCE SUMMARY")
    print("=" * 60)
    
    df_results = pd.DataFrame(results)
    
    print(f"📊 Performance Results:")
    for _, row in df_results.iterrows():
        gpu_util_str = f" (GPU: {row['gpu_utilization']:.1f}%)" if gpu_available else ""
        print(f"   {row['size']:>6,} tweets: {row['speed']:>8.1f} tweets/sec{gpu_util_str}")
    
    if len(results) > 1:
        max_speed = df_results['speed'].max()
        avg_speed = df_results['speed'].mean()
        max_gpu_util = df_results['gpu_utilization'].max() if gpu_available else 0
        
        print(f"\n🚀 Peak Performance: {max_speed:.1f} tweets/sec")
        print(f"📊 Average Speed: {avg_speed:.1f} tweets/sec")
        
        if gpu_available:
            print(f"🎯 Maximum GPU Utilization: {max_gpu_util:.1f}%")
            
            if max_gpu_util >= 90:
                print("✅ AGGRESSIVE GPU OPTIMIZATION SUCCESS: 90-100% utilization achieved!")
            elif max_gpu_util >= 70:
                print("⚠️  Good GPU utilization, but room for improvement")
            else:
                print("❌ Low GPU utilization - check GPU settings")
        
        # Calculate speedup vs smallest test
        if len(results) >= 2:
            base_speed = results[0]['speed']
            peak_speed = max_speed
            speedup = peak_speed / base_speed if base_speed > 0 else 1
            print(f"📈 Speed Improvement: {speedup:.1f}x faster at scale")
    
    print("\n🎯 AGGRESSIVE GPU TEST COMPLETE!")
    
    return results

if __name__ == "__main__":
    test_aggressive_gpu_performance()
