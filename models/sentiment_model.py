"""
Sentiment analysis model using HuggingFace transformers.
Uses the powerful cardiffnlp/twitter-roberta-base-sentiment model.
AGGRESSIVE GPU OPTIMIZATION - Uses 90-100% of available GPU memory when available.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
import time
from typing import List, Union, Dict
import warnings
import streamlit as st
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

def set_aggressive_gpu_settings():
    """Configure PyTorch for aggressive GPU memory usage."""
    if torch.cuda.is_available():
        # Enable aggressive GPU memory allocation
        torch.cuda.empty_cache()
        
        # Set memory fraction to use almost all available GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Use 95% of available GPU memory (leaving 5% for system)
        memory_fraction = 0.95
        
        # Enable memory growth and set cache allocator
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable aggressive CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Set aggressive tensor core usage
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logging.info(f"🚀 AGGRESSIVE GPU MODE: Using {memory_fraction*100}% of GPU memory")
        logging.info(f"💾 Total GPU Memory: {total_memory/1024**3:.1f}GB")
        logging.info(f"🎯 Allocated Memory: {total_memory * memory_fraction/1024**3:.1f}GB")

@st.cache_resource
def load_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Load and cache the sentiment analysis model with AGGRESSIVE GPU optimization.
    Uses 90-100% of available GPU memory for maximum performance.
    """
    try:
        logging.info(f"Loading sentiment model: {model_name}")
        
        # Check GPU availability and set aggressive settings
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"🚀 GPU detected: {gpu_name}")
            logging.info(f"💾 GPU memory: {gpu_memory:.1f}GB")
            
            # Set aggressive GPU optimization
            set_aggressive_gpu_settings()
            device = 0
        else:
            logging.info("💻 Using CPU (GPU not available)")
            device = -1
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to GPU if available with aggressive settings
        if device == 0:
            model = model.cuda()
            # Set model to evaluation mode for better GPU utilization
            model.eval()
            
            # Enable aggressive GPU optimizations for the model
            with torch.cuda.device(0):
                # Warm up GPU with dummy input for optimal memory allocation
                dummy_input = tokenizer("warmup text", return_tensors="pt", padding=True, truncation=True)
                dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
                with torch.no_grad():
                    _ = model(**dummy_input)
                
            logging.info("✅ Model loaded on GPU with AGGRESSIVE optimization")
            
            # Show GPU memory usage
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            logging.info(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        # Determine optimal batch size based on GPU memory and aggressive settings
        if device == 0:
            if gpu_memory >= 6:  # RTX 3050 6GB or better
                batch_size = 128  # AGGRESSIVE: 4x larger batch size
            elif gpu_memory >= 4:  # RTX 3050 4GB
                batch_size = 96   # AGGRESSIVE: 3x larger batch size
            else:
                batch_size = 64   # AGGRESSIVE: 2x larger batch size
            logging.info(f"🎯 AGGRESSIVE batch size: {batch_size}")
        else:
            batch_size = 8  # Conservative for CPU
        
        # Create optimized pipeline for AGGRESSIVE GPU usage
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            batch_size=batch_size,  # AGGRESSIVE batch size
            max_length=512,
            truncation=True,
            padding=True
        )
        
        logging.info("✅ Sentiment model loaded successfully with AGGRESSIVE GPU optimization!")
        return classifier
        
    except Exception as e:
        logging.error(f"❌ Error loading sentiment model: {str(e)}")
        # Fallback to a simpler model if the main one fails
        try:
            logging.info("Loading fallback sentiment model...")
            fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
            
            classifier = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=64 if torch.cuda.is_available() else 8  # AGGRESSIVE fallback
            )
            
            logging.info("✅ Fallback model loaded successfully!")
            return classifier
            
        except Exception as e2:
            logging.error(f"❌ Error loading fallback model: {str(e2)}")
            return None

class SentimentAnalyzer:
    """
    Sentiment analyzer using RoBERTa model trained specifically on Twitter data.
    This is much more accurate than traditional LSTM models.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.classifier = load_sentiment_model(model_name)  # Use cached model loading
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def predict_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with sentiment label and confidence
        """
        if not self.classifier:
            return {"label": "NEUTRAL", "score": 0.5}
        
        if not text or not isinstance(text, str):
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            # Clean text for analysis
            text = text.strip()
            if len(text) == 0:
                return {"label": "NEUTRAL", "score": 0.5}
            
            # Get prediction
            result = self.classifier(text)
            
            # Handle different model output formats
            if isinstance(result[0], list):
                # Model returns all scores (like RoBERTa)
                scores = result[0]
                best_result = max(scores, key=lambda x: x['score'])
            else:
                # Model returns single best result (like DistilBERT)
                best_result = result[0]
            
            # Normalize label names
            label = best_result['label'].upper()
            if label in ['LABEL_0', 'NEGATIVE']:
                label = 'NEGATIVE'
            elif label in ['LABEL_1', 'NEUTRAL']:
                label = 'NEUTRAL'
            elif label in ['LABEL_2', 'POSITIVE']:
                label = 'POSITIVE'
            
            return {
                "label": label,
                "score": float(best_result['score'])
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting sentiment: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def predict_batch(self, texts: List[str], batch_size: int = None) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for a batch of texts with AGGRESSIVE GPU optimization.
        Uses 90-100% of GPU memory for maximum performance.
        
        Args:
            texts (List[str]): List of texts to analyze
            batch_size (int): Processing batch size (auto-optimized for AGGRESSIVE GPU usage)
            
        Returns:
            List[dict]: List of sentiment predictions
        """
        if not self.classifier:
            return [{"label": "NEUTRAL", "score": 0.5}] * len(texts)
        
        total_texts = len(texts)
        
        # Set start time for progress tracking
        self._start_time = time.time()
        
        # AGGRESSIVE auto-optimization of batch size based on hardware
        if batch_size is None:
            if torch.cuda.is_available():
                # AGGRESSIVE GPU optimization for maximum memory usage
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                if gpu_memory >= 6:  # RTX 3050 6GB or better
                    batch_size = 256  # AGGRESSIVE: 8x larger batches
                    self.logger.info(f"🚀 AGGRESSIVE GPU (6GB+): {batch_size} tweets per batch")
                elif gpu_memory >= 4:  # RTX 3050 4GB
                    batch_size = 192  # AGGRESSIVE: 6x larger batches
                    self.logger.info(f"🚀 AGGRESSIVE GPU (4GB+): {batch_size} tweets per batch")
                elif gpu_memory >= 2:  # Entry level GPU
                    batch_size = 128  # AGGRESSIVE: 4x larger batches
                    self.logger.info(f"🚀 AGGRESSIVE GPU (2GB+): {batch_size} tweets per batch")
                else:
                    batch_size = 64   # AGGRESSIVE: 2x larger batches
                    self.logger.info(f"🚀 AGGRESSIVE GPU (<2GB): {batch_size} tweets per batch")
                
                # Force GPU memory cleanup before processing
                torch.cuda.empty_cache()
                gc.collect()
                
                # Show current GPU utilization
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                utilization = (allocated / total_memory) * 100
                self.logger.info(f"� GPU Memory Usage: {allocated:.2f}GB / {total_memory:.1f}GB ({utilization:.1f}%)")
                
            else:
                batch_size = 16  # Slightly larger for CPU
                self.logger.info(f"💻 CPU OPTIMIZED: {batch_size} tweets per batch")
        
        if total_texts > 1000:
            self.logger.info(f"🚀 AGGRESSIVE PROCESSING: {total_texts:,} texts with maximum GPU utilization...")
        
        results = []
        
        # AGGRESSIVE GPU-optimized batch processing
        for start_idx in range(0, total_texts, batch_size):
            end_idx = min(start_idx + batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # Clean and prepare texts for aggressive batch processing
            clean_texts = [text.strip() if text else "" for text in batch_texts]
            non_empty_texts = [text for text in clean_texts if text]
            
            if non_empty_texts:
                try:
                    # AGGRESSIVE GPU batch inference with memory optimization
                    if torch.cuda.is_available() and len(non_empty_texts) > 1:
                        # Pre-allocate GPU memory for the batch
                        with torch.cuda.device(0):
                            # AGGRESSIVE GPU batch processing
                            batch_predictions = self.classifier(non_empty_texts)
                            
                            # Force memory cleanup after each batch
                            torch.cuda.empty_cache()
                        
                        # Process results efficiently
                        batch_results = []
                        prediction_idx = 0
                        
                        for text in clean_texts:
                            if text:
                                if isinstance(batch_predictions[prediction_idx], list):
                                    # Handle models that return all scores
                                    best_pred = max(batch_predictions[prediction_idx], key=lambda x: x['score'])
                                else:
                                    best_pred = batch_predictions[prediction_idx]
                                
                                # Normalize labels
                                label = self.normalize_label(best_pred['label'])
                                batch_results.append({
                                    "label": label,
                                    "score": float(best_pred['score'])
                                })
                                prediction_idx += 1
                            else:
                                batch_results.append({"label": "NEUTRAL", "score": 0.5})
                    else:
                        # Fallback to individual processing for small batches
                        batch_results = []
                        for text in clean_texts:
                            result = self.predict_sentiment(text) if text else {"label": "NEUTRAL", "score": 0.5}
                            batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"AGGRESSIVE batch processing error: {e}")
                    # Fallback to smaller batches if memory issues
                    if torch.cuda.is_available() and "out of memory" in str(e).lower():
                        self.logger.warning("⚠️ GPU memory exceeded, falling back to smaller batches...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Reduce batch size and retry
                        smaller_batch_size = batch_size // 2
                        batch_results = []
                        for mini_start in range(0, len(clean_texts), smaller_batch_size):
                            mini_end = min(mini_start + smaller_batch_size, len(clean_texts))
                            mini_batch = clean_texts[mini_start:mini_end]
                            mini_non_empty = [t for t in mini_batch if t]
                            
                            if mini_non_empty:
                                mini_predictions = self.classifier(mini_non_empty)
                                mini_results = []
                                pred_idx = 0
                                
                                for text in mini_batch:
                                    if text:
                                        if isinstance(mini_predictions[pred_idx], list):
                                            best_pred = max(mini_predictions[pred_idx], key=lambda x: x['score'])
                                        else:
                                            best_pred = mini_predictions[pred_idx]
                                        
                                        label = self.normalize_label(best_pred['label'])
                                        mini_results.append({
                                            "label": label,
                                            "score": float(best_pred['score'])
                                        })
                                        pred_idx += 1
                                    else:
                                        mini_results.append({"label": "NEUTRAL", "score": 0.5})
                                
                                batch_results.extend(mini_results)
                    else:
                        # Non-memory related error, fallback to individual processing
                        batch_results = []
                        for text in clean_texts:
                            result = self.predict_sentiment(text) if text else {"label": "NEUTRAL", "score": 0.5}
                            batch_results.append(result)
            else:
                batch_results = [{"label": "NEUTRAL", "score": 0.5}] * len(batch_texts)
            
            results.extend(batch_results)
            
            # Show AGGRESSIVE progress with GPU utilization stats
            if total_texts > 1000 and end_idx % (batch_size * 2) == 0:  # More frequent updates for aggressive mode
                progress = (end_idx / total_texts) * 100
                elapsed_time = time.time() - getattr(self, '_start_time', time.time())
                speed = end_idx / (elapsed_time + 0.001)
                remaining_texts = total_texts - end_idx
                # Show AGGRESSIVE GPU utilization stats
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    utilization = (allocated / total_memory) * 100
                    self.logger.info(f"🎯 GPU Memory: {allocated:.2f}GB / {total_memory:.1f}GB ({utilization:.1f}% utilization)")
                
                eta_seconds = remaining_texts / (speed + 0.001)
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
                
                self.logger.info(f"📊 AGGRESSIVE Progress: {end_idx:,} / {total_texts:,} ({progress:.1f}%) - Speed: {speed:.1f} tweets/sec - ETA: {eta_str}")
                
                # Aggressive memory management
                if end_idx % (batch_size * 10) == 0:  # More frequent cleanup in aggressive mode
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure all GPU operations complete
        
        if total_texts > 1000:
            elapsed_time = time.time() - getattr(self, '_start_time', time.time())
            speed = total_texts / (elapsed_time + 0.001)
            
            if torch.cuda.is_available():
                final_allocated = torch.cuda.memory_allocated(0) / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                final_utilization = (final_allocated / total_memory) * 100
                self.logger.info(f"✅ AGGRESSIVE PROCESSING COMPLETE: {total_texts:,} texts in {elapsed_time:.1f}s")
                self.logger.info(f"🚀 Average Speed: {speed:.1f} tweets/sec with {final_utilization:.1f}% GPU utilization")
            else:
                self.logger.info(f"✅ CPU Processing complete: {total_texts:,} texts in {elapsed_time:.1f}s (avg: {speed:.1f} tweets/sec)")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of tweets.
        
        Args:
            df (pd.DataFrame): DataFrame containing tweets
            text_column (str): Name of the column containing text
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        if df.empty or text_column not in df.columns:
            return df
        
        self.logger.info(f"Analyzing sentiment for {len(df)} tweets...")
        
        # Get sentiment predictions
        texts = df[text_column].fillna("").astype(str).tolist()
        predictions = self.predict_batch(texts)
        
        # Add results to dataframe
        df['sentiment'] = [pred['label'] for pred in predictions]
        df['sentiment_score'] = [pred['score'] for pred in predictions]
        
        # Add sentiment category for easier filtering
        df['sentiment_category'] = df['sentiment'].map({
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral'
        })
        
        self.logger.info("✅ Sentiment analysis completed!")
        
        return df
    
    def normalize_label(self, label: str) -> str:
        """
        Normalize sentiment labels from different models to consistent format.
        
        Args:
            label (str): Raw label from model
            
        Returns:
            str: Normalized label (POSITIVE, NEGATIVE, NEUTRAL)
        """
        label = label.upper()
        if label in ['LABEL_0', 'NEGATIVE']:
            return 'NEGATIVE'
        elif label in ['LABEL_1', 'NEUTRAL']:
            return 'NEUTRAL'
        elif label in ['LABEL_2', 'POSITIVE']:
            return 'POSITIVE'
        else:
            return label  # Return as-is if already normalized
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics of sentiment analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            dict: Summary statistics
        """
        if 'sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['sentiment'].value_counts()
        total_tweets = len(df)
        
        summary = {
            'total_tweets': total_tweets,
            'positive_count': sentiment_counts.get('POSITIVE', 0),
            'negative_count': sentiment_counts.get('NEGATIVE', 0),
            'neutral_count': sentiment_counts.get('NEUTRAL', 0),
            'positive_percentage': round((sentiment_counts.get('POSITIVE', 0) / total_tweets) * 100, 2),
            'negative_percentage': round((sentiment_counts.get('NEGATIVE', 0) / total_tweets) * 100, 2),
            'neutral_percentage': round((sentiment_counts.get('NEUTRAL', 0) / total_tweets) * 100, 2)
        }
        
        return summary

# Global instance for easy import
analyzer = SentimentAnalyzer()

def analyze_sentiment(text: str) -> str:
    """
    Simple function to get sentiment of a text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Sentiment label (POSITIVE, NEGATIVE, NEUTRAL)
    """
    result = analyzer.predict_sentiment(text)
    return result['label']

def analyze_tweets_sentiment(df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
    """
    Analyze sentiment for a DataFrame of tweets.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets
        text_column (str): Name of the column containing text
        
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
    """
    return analyzer.analyze_dataframe(df, text_column)

def get_gpu_memory_info() -> Dict[str, Union[str, float]]:
    """
    Get detailed GPU memory information for aggressive optimization monitoring.
    
    Returns:
        dict: GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {"status": "No GPU available"}
    
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    
    # Memory information
    total_memory = properties.total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3
    free_memory = total_memory - allocated_memory
    
    utilization_percentage = (allocated_memory / total_memory) * 100
    
    return {
        "device_name": properties.name,
        "total_memory_gb": round(total_memory, 2),
        "allocated_memory_gb": round(allocated_memory, 2),
        "cached_memory_gb": round(cached_memory, 2),
        "free_memory_gb": round(free_memory, 2),
        "utilization_percentage": round(utilization_percentage, 1),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }

def optimize_gpu_for_aggressive_processing():
    """
    Apply aggressive GPU optimizations for maximum performance.
    Call this before processing large datasets.
    """
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set aggressive GPU settings
        set_aggressive_gpu_settings()
        
        # Log optimization status
        gpu_info = get_gpu_memory_info()
        logging.info("🚀 AGGRESSIVE GPU OPTIMIZATION APPLIED:")
        logging.info(f"   Device: {gpu_info['device_name']}")
        logging.info(f"   Total Memory: {gpu_info['total_memory_gb']}GB")
        logging.info(f"   Current Utilization: {gpu_info['utilization_percentage']}%")
        logging.info(f"   CUDA Version: {gpu_info['cuda_version']}")
        
        return True
    else:
        logging.info("💻 No GPU available - using CPU optimization")
        return False

def force_gpu_memory_cleanup():
    """
    Aggressively clean up GPU memory.
    Use this between processing batches or when encountering memory issues.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Show memory status after cleanup
        gpu_info = get_gpu_memory_info()
        logging.info(f"🧹 GPU Memory Cleanup - Current utilization: {gpu_info['utilization_percentage']}%")

# Initialize aggressive GPU settings on module import
if torch.cuda.is_available():
    optimize_gpu_for_aggressive_processing()
