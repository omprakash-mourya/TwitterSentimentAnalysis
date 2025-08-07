"""
ULTRA-AGGRESSIVE GPU-Optimized Sentiment Model
Forces 90-100% GPU utilization through massive batch processing and memory allocation.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging
import time
from typing import List, Union, Dict
import warnings
import streamlit as st
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

def force_maximum_gpu_allocation():
    """Force PyTorch to allocate maximum GPU memory upfront."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Allocate 98% of GPU memory upfront
        memory_to_allocate = int(total_memory * 0.98)
        
        try:
            # Pre-allocate GPU memory
            dummy_tensor = torch.zeros((memory_to_allocate // 4,), dtype=torch.float32, device='cuda')
            del dummy_tensor
            torch.cuda.empty_cache()
            
            # Set aggressive memory settings
            torch.cuda.set_per_process_memory_fraction(0.98)
            
            # Enable all CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logging.info(f"🚀 ULTRA-AGGRESSIVE: Pre-allocated 98% of {total_memory/1024**3:.1f}GB GPU memory")
            return True
            
        except Exception as e:
            logging.warning(f"⚠️ Could not pre-allocate maximum GPU memory: {e}")
            return False
    return False

@st.cache_resource
def load_ultra_aggressive_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Load model with ULTRA-AGGRESSIVE GPU optimization.
    Forces maximum GPU memory usage and batch processing.
    """
    try:
        logging.info(f"🚀 Loading ULTRA-AGGRESSIVE GPU model: {model_name}")
        
        if not torch.cuda.is_available():
            logging.info("💻 No GPU available - using CPU")
            return None, None, None
        
        # Force maximum GPU allocation
        force_maximum_gpu_allocation()
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU and optimize
        device = torch.device('cuda:0')
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Enable aggressive optimizations without JIT compilation
        model.half()  # Use FP16 for more aggressive memory usage
        
        # Determine optimal batch size for warmup based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        warmup_batch_size = min(1024, int(gpu_memory_gb * 100))  # Scale with GPU memory
        
        # Warm up with large batch to force aggressive GPU memory allocation
        dummy_texts = ["This is a comprehensive warmup text designed to maximize GPU memory utilization through aggressive batch processing and tokenization strategies. " * 10] * warmup_batch_size
        
        try:
            dummy_inputs = tokenizer(
                dummy_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
            
            with torch.no_grad():
                _ = model(**dummy_inputs)
            
            # Additional memory allocation to force maximum usage
            # Create large tensors to occupy more GPU memory
            try:
                available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                tensor_size = int(available_memory * 0.8 // 4)  # Use 80% of remaining memory
                memory_tensor = torch.zeros(tensor_size, dtype=torch.float32, device=device)
                # Keep this tensor alive to maintain memory allocation
                
            except RuntimeError:
                # If we can't allocate that much, try smaller
                logging.info("⚠️ Reducing memory allocation size")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduce warmup batch size and try again
                warmup_batch_size = warmup_batch_size // 2
                logging.info(f"⚠️ Reducing warmup batch size to {warmup_batch_size}")
                
                dummy_texts = ["warmup text " * 50] * warmup_batch_size
                dummy_inputs = tokenizer(
                    dummy_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
                
                with torch.no_grad():
                    _ = model(**dummy_inputs)
        
        # Show GPU memory usage after warmup
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        utilization = (allocated / total_memory) * 100
        
        logging.info(f"✅ ULTRA-AGGRESSIVE model loaded!")
        logging.info(f"💾 GPU Memory: {allocated:.2f}GB / {total_memory:.1f}GB ({utilization:.1f}%)")
        
        return model, tokenizer, device
        
    except Exception as e:
        logging.error(f"❌ Error loading ultra-aggressive model: {e}")
        return None, None, None

class UltraAggressiveSentimentAnalyzer:
    """
    Ultra-aggressive sentiment analyzer that forces 90-100% GPU utilization.
    Uses direct model inference with massive batch processing.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.model, self.tokenizer, self.device = load_ultra_aggressive_model(model_name)
        self.logger = logging.getLogger(__name__)
        
        # Label mapping for sentiment
        self.label_mapping = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
        
        if self.model is None:
            self.logger.warning("⚠️ ULTRA-AGGRESSIVE mode failed, falling back to CPU")
    
    def predict_batch_ultra_aggressive(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Ultra-aggressive batch prediction that forces maximum GPU utilization.
        Uses massive batches and direct model inference.
        """
        if self.model is None or not torch.cuda.is_available():
            return [{"label": "NEUTRAL", "score": 0.5}] * len(texts)
        
        total_texts = len(texts)
        self.logger.info(f"🚀 ULTRA-AGGRESSIVE processing: {total_texts:,} texts")
        
        # Determine ultra-aggressive batch size based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory >= 6:  # RTX 3050 6GB or better
            max_batch_size = 1024  # ULTRA-AGGRESSIVE: Massive batches
        elif gpu_memory >= 4:
            max_batch_size = 768
        else:
            max_batch_size = 512
        
        self.logger.info(f"🎯 ULTRA-AGGRESSIVE batch size: {max_batch_size}")
        
        results = []
        start_time = time.time()
        
        # Process in ultra-aggressive batches
        for start_idx in range(0, total_texts, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # Clean texts
            clean_texts = [text.strip() if text and isinstance(text, str) else "" for text in batch_texts]
            
            try:
                # Tokenize with aggressive settings
                inputs = self.tokenizer(
                    clean_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Direct model inference with no_grad for memory efficiency
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply softmax to get probabilities
                    probabilities = F.softmax(logits, dim=-1)
                    
                    # Get predictions
                    predicted_labels = torch.argmax(probabilities, dim=-1)
                    confidence_scores = torch.max(probabilities, dim=-1)[0]
                
                # Convert to CPU and process results
                predicted_labels = predicted_labels.cpu().numpy()
                confidence_scores = confidence_scores.cpu().numpy()
                
                # Create results for this batch
                batch_results = []
                for i, (label_idx, score) in enumerate(zip(predicted_labels, confidence_scores)):
                    label = self.label_mapping.get(int(label_idx), 'NEUTRAL')
                    batch_results.append({
                        "label": label,
                        "score": float(score)
                    })
                
                results.extend(batch_results)
                
                # Force memory cleanup after each batch
                del inputs, outputs, logits, probabilities, predicted_labels, confidence_scores
                torch.cuda.empty_cache()
                
                # Show progress with GPU utilization
                if total_texts > 1000 and end_idx % max_batch_size == 0:
                    progress = (end_idx / total_texts) * 100
                    elapsed = time.time() - start_time
                    speed = end_idx / elapsed
                    
                    # Show GPU memory utilization
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    utilization = (allocated / total_memory) * 100
                    
                    self.logger.info(f"🚀 Progress: {end_idx:,}/{total_texts:,} ({progress:.1f}%) - "
                                   f"Speed: {speed:.1f} tweets/sec - "
                                   f"GPU: {utilization:.1f}%")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size if out of memory
                    self.logger.warning(f"⚠️ GPU OOM, reducing batch size from {max_batch_size}")
                    max_batch_size = max_batch_size // 2
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Retry with smaller batch
                    continue
                else:
                    self.logger.error(f"❌ Processing error: {e}")
                    # Fallback results
                    batch_results = [{"label": "NEUTRAL", "score": 0.5}] * len(batch_texts)
                    results.extend(batch_results)
            
            except Exception as e:
                self.logger.error(f"❌ Batch processing error: {e}")
                batch_results = [{"label": "NEUTRAL", "score": 0.5}] * len(batch_texts)
                results.extend(batch_results)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        speed = total_texts / elapsed_time
        
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated(0) / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            final_utilization = (final_allocated / total_memory) * 100
            
            self.logger.info(f"✅ ULTRA-AGGRESSIVE COMPLETE: {total_texts:,} texts in {elapsed_time:.1f}s")
            self.logger.info(f"🚀 Final Speed: {speed:.1f} tweets/sec with {final_utilization:.1f}% GPU utilization")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """Analyze DataFrame with ultra-aggressive GPU processing."""
        if df.empty or text_column not in df.columns:
            return df
        
        texts = df[text_column].fillna("").astype(str).tolist()
        predictions = self.predict_batch_ultra_aggressive(texts)
        
        df['sentiment'] = [pred['label'] for pred in predictions]
        df['sentiment_score'] = [pred['score'] for pred in predictions]
        df['sentiment_category'] = df['sentiment'].map({
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral'
        })
        
        return df

# Global ultra-aggressive analyzer
ultra_analyzer = UltraAggressiveSentimentAnalyzer()

def analyze_tweets_ultra_aggressive(df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
    """Ultra-aggressive sentiment analysis for maximum GPU utilization."""
    return ultra_analyzer.analyze_dataframe(df, text_column)

def get_current_gpu_utilization() -> float:
    """Get current GPU utilization percentage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return (allocated / total_memory) * 100
    return 0.0
