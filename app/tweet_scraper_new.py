"""
Improved Tweet scraper using snscrape - No API keys required!
Combines efficient scraping techniques from all three projects.
"""

import pandas as pd
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import logging
import time
import random

class TwitterScraper:
    """
    Twitter scraper using snscrape for real-time tweet collection.
    No API keys required!
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for scraper activities"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def scrape_tweets(self, query, max_tweets=100, days_back=7):
        """
        Scrape tweets for a given query using snscrape.
        
        Args:
            query (str): Search query/keyword
            max_tweets (int): Maximum number of tweets to scrape
            days_back (int): How many days back to search
            
        Returns:
            pd.DataFrame: DataFrame containing scraped tweets
        """
        tweets_list = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for snscrape
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Build search query with filters
            search_query = f"{query} since:{start_date_str} until:{end_date_str} lang:en -filter:retweets"
            
            self.logger.info(f"Scraping tweets for query: {search_query}")
            
            # Scrape tweets
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
                if i >= max_tweets:
                    break
                    
                # Extract tweet data
                tweet_data = {
                    'id': tweet.id,
                    'date': tweet.date,
                    'content': tweet.content,
                    'user': tweet.user.username,
                    'user_followers': getattr(tweet.user, 'followersCount', 0),
                    'user_verified': getattr(tweet.user, 'verified', False),
                    'like_count': tweet.likeCount or 0,
                    'retweet_count': tweet.retweetCount or 0,
                    'reply_count': tweet.replyCount or 0,
                    'language': getattr(tweet, 'lang', 'en'),
                    'source': getattr(tweet, 'sourceLabel', 'Unknown'),
                    'query_term': query,
                    'url': tweet.url
                }
                
                tweets_list.append(tweet_data)
                
                # Add small delay to be respectful
                time.sleep(random.uniform(0.1, 0.3))
            
            # Create DataFrame
            df = pd.DataFrame(tweets_list)
            
            if not df.empty:
                # Sort by date (newest first)
                df = df.sort_values('date', ascending=False)
                
                # Add timestamp for when the data was scraped
                df['scraped_at'] = datetime.now()
                
                self.logger.info(f"Successfully scraped {len(df)} tweets for '{query}'")
            else:
                self.logger.warning(f"No tweets found for query: {query}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error scraping tweets: {str(e)}")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=[
                'id', 'date', 'content', 'user', 'user_followers', 'user_verified',
                'like_count', 'retweet_count', 'reply_count', 'language', 'source',
                'query_term', 'url', 'scraped_at'
            ])

# Global instance for easy import
scraper = TwitterScraper()

def get_tweets(query, count=100):
    """
    Simple function to get tweets for a query.
    
    Args:
        query (str): Search term
        count (int): Number of tweets to fetch
        
    Returns:
        pd.DataFrame: Tweets dataframe
    """
    return scraper.scrape_tweets(query, max_tweets=count)

def scrape_tweets(query, max_tweets=100):
    """
    Legacy function for backward compatibility
    """
    return get_tweets(query, max_tweets)
