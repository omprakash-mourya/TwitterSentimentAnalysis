"""
Text cleaning utility for tweet preprocessing.
Combines the best preprocessing techniques from all three projects.
"""

import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TweetCleaner:
    """
    Text cleaning class that combines techniques from all three projects.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Additional custom stopwords
        self.stop_words.update(['rt', 'via', 'amp', 'http', 'https', 'www'])
        
    def clean_text(self, text):
        """
        Clean tweet text using combined techniques from all projects.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned tweet text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs (from Real-Time project)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', '', text)
        
        # Remove mentions and hashtags (from tweet-sense project)
        text = re.sub(r'(@|#)\w+', '', text)
        text = re.sub(r'@\S+', '', text)
        
        # Remove retweet markers
        text = re.sub(r'\brt\b', '', text)
        
        # Remove numbers (from tweet-sense project)
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_for_sentiment(self, text):
        """
        Preprocess text specifically for sentiment analysis.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Preprocessed text ready for sentiment analysis
        """
        # Basic cleaning
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return ""
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token.lower() not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def preprocess_for_wordcloud(self, text):
        """
        Preprocess text for word cloud generation.
        Keep more words but clean appropriately.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Text ready for word cloud
        """
        # Basic cleaning but keep more words
        text = self.clean_text(text)
        
        # Remove only basic stopwords, keep meaningful words
        basic_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = [
            token for token in tokens 
            if token.lower() not in basic_stopwords 
            and len(token) > 2 
            and not token.startswith(('http', '@', '#'))
        ]
        
        return ' '.join(filtered_tokens)

def clean_tweet(text):
    """
    Clean tweet text by removing mentions, links, hashtags, and special characters
    Legacy function for backward compatibility
    """
    cleaner = TweetCleaner()
    return cleaner.clean_text(text)

def preprocess_tweets(tweets_df):
    """
    Preprocess a dataframe of tweets
    """
    if 'content' in tweets_df.columns:
        tweets_df['cleaned_content'] = tweets_df['content'].apply(clean_tweet)
    elif 'text' in tweets_df.columns:
        tweets_df['cleaned_content'] = tweets_df['text'].apply(clean_tweet)
    else:
        # If no content or text column, create a dummy one
        tweets_df['cleaned_content'] = ""
    
    # Remove empty tweets after cleaning
    tweets_df = tweets_df[tweets_df['cleaned_content'].str.len() > 10]
    
    return tweets_df

# Global instance for easy import
cleaner = TweetCleaner()
