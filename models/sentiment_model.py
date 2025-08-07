"""
Sentiment analysis model using HuggingFace transformers.
Uses the powerful cardiffnlp/twitter-roberta-base-sentiment model.
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

# Suppress warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Load and cache the sentiment analysis model with GPU optimization.
    Optimized for RTX 3050 6GB GPU acceleration.
    """
    try:
        logging.info(f"Loading sentiment model: {model_name}")
        
        # Check GPU availability and memory
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"🚀 GPU detected: {gpu_name}")
            logging.info(f"💾 GPU memory: {gpu_memory:.1f}GB")
            device = 0
        else:
            logging.info("💻 Using CPU (GPU not available)")
            device = -1
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to GPU if available
        if device == 0:
            model = model.cuda()
            logging.info("✅ Model loaded on GPU")
        
        # Create optimized pipeline for GPU
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            batch_size=16 if device == 0 else 1,  # Larger batch for GPU
            max_length=512,
            truncation=True,
            padding=True
        )
        
        logging.info("✅ Sentiment model loaded successfully!")
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
                device=0 if torch.cuda.is_available() else -1
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
        Predict sentiment for a batch of texts with GPU-optimized processing.
        Automatically adjusts batch size based on available hardware.
        
        Args:
            texts (List[str]): List of texts to analyze
            batch_size (int): Processing batch size (auto-optimized if None)
            
        Returns:
            List[dict]: List of sentiment predictions
        """
        if not self.classifier:
            return [{"label": "NEUTRAL", "score": 0.5}] * len(texts)
        
        total_texts = len(texts)
        
        # Set start time for progress tracking
        self._start_time = time.time()
        
        # Auto-optimize batch size based on hardware
        if batch_size is None:
            if torch.cuda.is_available():
                # GPU optimization for RTX 3050 6GB
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 6:  # RTX 3050 or better
                    batch_size = 32  # Larger batches for GPU
                else:
                    batch_size = 16
                self.logger.info(f"🚀 GPU batch processing: {batch_size} tweets per batch")
            else:
                batch_size = 8  # Conservative for CPU
                self.logger.info(f"💻 CPU batch processing: {batch_size} tweets per batch")
        
        if total_texts > 1000:
            self.logger.info(f"🚀 Processing {total_texts:,} texts with GPU-optimized batching...")
        
        results = []
        
        # GPU-optimized batch processing
        for start_idx in range(0, total_texts, batch_size):
            end_idx = min(start_idx + batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # Clean and prepare texts for batch processing
            clean_texts = [text.strip() if text else "" for text in batch_texts]
            non_empty_texts = [text for text in clean_texts if text]
            
            if non_empty_texts:
                try:
                    # Use pipeline's built-in batch processing for GPU acceleration
                    if torch.cuda.is_available() and len(non_empty_texts) > 1:
                        # GPU batch inference - much faster!
                        batch_predictions = self.classifier(non_empty_texts)
                        
                        # Process results
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
                        # Fallback to individual processing
                        batch_results = []
                        for text in clean_texts:
                            result = self.predict_sentiment(text) if text else {"label": "NEUTRAL", "score": 0.5}
                            batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    # Fallback to individual processing
                    batch_results = []
                    for text in clean_texts:
                        result = self.predict_sentiment(text) if text else {"label": "NEUTRAL", "score": 0.5}
                        batch_results.append(result)
            else:
                batch_results = [{"label": "NEUTRAL", "score": 0.5}] * len(batch_texts)
            
            results.extend(batch_results)
            
            # Show progress for large datasets with connection keepalive
            if total_texts > 1000 and end_idx % (batch_size * 5) == 0:  # More frequent updates
                progress = (end_idx / total_texts) * 100
                elapsed_time = time.time() - getattr(self, '_start_time', time.time())
                speed = end_idx / (elapsed_time + 0.001)
                remaining_texts = total_texts - end_idx
                eta_seconds = remaining_texts / (speed + 0.001)
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
                
                self.logger.info(f"📊 Progress: {end_idx:,} / {total_texts:,} ({progress:.1f}%) - Speed: {speed:.1f} tweets/sec - ETA: {eta_str}")
                
                # Force garbage collection to manage memory
                if end_idx % (batch_size * 20) == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        if total_texts > 1000:
            elapsed_time = time.time() - getattr(self, '_start_time', time.time())
            speed = total_texts / (elapsed_time + 0.001)
            self.logger.info(f"✅ Completed processing {total_texts:,} texts with GPU acceleration in {elapsed_time:.1f}s (avg: {speed:.1f} tweets/sec)")
        
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
