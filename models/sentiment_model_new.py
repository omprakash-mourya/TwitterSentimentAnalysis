"""
Sentiment analysis model using HuggingFace transformers.
Uses the powerful cardiffnlp/twitter-roberta-base-sentiment model.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from typing import List, Union, Dict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class SentimentAnalyzer:
    """
    Sentiment analyzer using RoBERTa model trained specifically on Twitter data.
    This is much more accurate than traditional LSTM models.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained RoBERTa sentiment model"""
        try:
            self.logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easy inference
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            self.logger.info("✅ Sentiment model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading sentiment model: {str(e)}")
            # Fallback to a simpler model if the main one fails
            self.load_fallback_model()
    
    def load_fallback_model(self):
        """Load a fallback sentiment model if the main one fails"""
        try:
            self.logger.info("Loading fallback sentiment model...")
            fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
            
            self.classifier = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                device=0 if self.device == "cuda" else -1
            )
            
            self.logger.info("✅ Fallback model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading fallback model: {str(e)}")
            self.classifier = None
    
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
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[dict]: List of sentiment predictions
        """
        if not self.classifier:
            return [{"label": "NEUTRAL", "score": 0.5}] * len(texts)
        
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        
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
