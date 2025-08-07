"""
Test Ultra-Aggressive GPU Model for Maximum Utilization
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ultra_aggressive_model import UltraAggressiveSentimentAnalyzer, get_current_gpu_utilization
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ultra_aggressive_performance():
    """Test ultra-aggressive GPU utilization."""
    
    print("🔥 ULTRA-AGGRESSIVE GPU TEST")
    print("=" * 60)
    print("🎯 Target: Force 90-100% GPU memory utilization")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ No GPU available for ultra-aggressive testing")
        return
    
    # Initialize ultra-aggressive analyzer
    analyzer = UltraAggressiveSentimentAnalyzer()
    
    if analyzer.model is None:
        print("❌ Failed to initialize ultra-aggressive model")
        return
    
    # Test with increasingly large batches to force GPU utilization
    test_sizes = [1000, 5000, 10000, 25000, 50000]
    
    for size in test_sizes:
        print(f"\n🚀 ULTRA-AGGRESSIVE TEST: {size:,} tweets")
        
        # Generate test data with longer texts to increase memory usage
        test_texts = [
            f"This is a comprehensive test tweet number {i} designed to maximize GPU memory utilization. "
            f"The sentiment analysis model should use aggressive batch processing to achieve 90-100% GPU utilization. "
            f"This longer text content will require more GPU memory for tokenization and processing, "
            f"forcing the model to utilize maximum available GPU resources for optimal performance. "
            f"The ultra-aggressive optimization should demonstrate significant improvements in processing speed "
            f"while maintaining high accuracy in sentiment classification across positive, negative, and neutral categories."
            for i in range(size)
        ]
        
        # Show initial GPU utilization
        initial_gpu = get_current_gpu_utilization()
        print(f"   📊 Initial GPU Utilization: {initial_gpu:.1f}%")
        
        # Run ultra-aggressive processing
        start_time = time.time()
        predictions = analyzer.predict_batch_ultra_aggressive(test_texts)
        end_time = time.time()
        
        # Calculate performance metrics
        processing_time = end_time - start_time
        speed = size / processing_time
        
        # Show final GPU utilization
        final_gpu = get_current_gpu_utilization()
        peak_gpu = max(initial_gpu, final_gpu)  # Approximate peak
        
        # Analyze results
        sentiment_counts = {}
        for pred in predictions:
            label = pred['label']
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        
        print(f"   🎯 Peak GPU Utilization: {peak_gpu:.1f}%")
        print(f"   ⚡ Speed: {speed:.1f} tweets/sec")
        print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
        print(f"   📊 Sentiment Distribution: {sentiment_counts}")
        
        # Evaluation
        if peak_gpu >= 90:
            print(f"   ✅ ULTRA-AGGRESSIVE SUCCESS: {peak_gpu:.1f}% GPU utilization!")
        elif peak_gpu >= 70:
            print(f"   ⚠️  Good utilization: {peak_gpu:.1f}% (target: 90%+)")
        else:
            print(f"   ❌ Low utilization: {peak_gpu:.1f}% (needs optimization)")
        
        # Brief pause between tests
        time.sleep(3)
        
        # Force cleanup
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("🔥 ULTRA-AGGRESSIVE GPU TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_ultra_aggressive_performance()
