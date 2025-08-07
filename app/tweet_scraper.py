"""
Improved Tweet scraper using snscrape - No API keys required!
Combines efficient scraping techniques from all three projects.
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import random

# Try importing snscrape with fallback for compatibility
try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except Exception as e:
    SNSCRAPE_AVAILABLE = False
    print(f"⚠️ snscrape not available: {e}")
    print("📝 Using mock data for demonstration")

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
            max_tweets (int): Maximum number of tweets to scrape (can be 1M+)
            days_back (int): How many days back to search
            
        Returns:
            pd.DataFrame: DataFrame containing scraped tweets
        """
        tweets_list = []
        
        if not SNSCRAPE_AVAILABLE:
            # Return mock data for demonstration (scale to requested amount)
            return self.generate_mock_tweets(query, max_tweets)
        
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
    
    def generate_mock_tweets(self, query, max_tweets):
        """
        Generate mock tweets for demonstration when snscrape is not available.
        Can handle large volumes (1M+ tweets) efficiently.
        """
        import random
        
        # Show progress for large datasets
        if max_tweets > 1000:
            self.logger.info(f"🔄 Generating {max_tweets:,} mock tweets for '{query}' - this may take a moment...")
        
        # Mock tweet templates with varying sentiments
        positive_tweets = [
            f"I absolutely love {query}! It's amazing and brings so much joy to my life 😊",
            f"{query} is fantastic! Highly recommend it to everyone 👍",
            f"Having a great time with {query}. Best decision ever! 🎉",
            f"{query} exceeded all my expectations. Brilliant work! ⭐",
            f"So happy with {query}! It's exactly what I needed 💖",
            f"{query} changed my life for the better! Incredible experience 🚀",
            f"Outstanding results with {query}! Five stars all the way! ⭐⭐⭐⭐⭐",
            f"Can't stop talking about {query}! Everyone should try this 🎯"
        ]
        
        negative_tweets = [
            f"Really disappointed with {query}. Not what I expected 😞",
            f"{query} is not working as promised. Very frustrating 😠",
            f"Had a terrible experience with {query}. Would not recommend ❌",
            f"{query} is overrated and overpriced. Not worth it 👎",
            f"Regret getting involved with {query}. Poor quality 💔",
            f"Worst decision ever to try {query}. Complete waste of time 😡",
            f"{query} failed to deliver on promises. Very disappointing 📉",
            f"Stay away from {query}! Nothing but problems and issues ⚠️"
        ]
        
        neutral_tweets = [
            f"Just tried {query}. It's okay, nothing special though 🤷",
            f"{query} has both pros and cons. Mixed feelings about it",
            f"Heard about {query} from a friend. Might give it a try",
            f"{query} seems interesting. Need more time to evaluate",
            f"People talking about {query} everywhere. Wonder what the hype is about",
            f"Currently testing {query}. Will update with results later 📊",
            f"{query} is trending but I'm not sure why. Looks average to me 🤔",
            f"Some say {query} is good, others disagree. Hard to tell 🤨"
        ]
        
        all_templates = positive_tweets + negative_tweets + neutral_tweets
        
        # Generate tweets efficiently for large datasets
        tweets_data = []
        batch_size = 10000  # Process in batches for memory efficiency
        
        for batch_start in range(0, max_tweets, batch_size):
            batch_end = min(batch_start + batch_size, max_tweets)
            batch_tweets = []
            
            for i in range(batch_start, batch_end):
                tweet_template = random.choice(all_templates)
                batch_tweets.append({
                    'date': datetime.now() - timedelta(days=random.randint(0, 7), 
                                                      hours=random.randint(0, 23),
                                                      minutes=random.randint(0, 59)),
                    'tweet_id': f"mock_{i}_{random.randint(10000, 99999)}",
                    'content': tweet_template,
                    'username': f"user_{random.randint(1000, 9999)}",
                    'like_count': random.randint(0, 10000),
                    'retweet_count': random.randint(0, 5000),
                    'reply_count': random.randint(0, 1000),
                    'language': 'en',
                    'source': 'Mock Twitter',
                    'query_term': query,
                    'url': f"https://twitter.com/mock/status/{i}",
                    'scraped_at': datetime.now()
                })
            
            tweets_data.extend(batch_tweets)
            
            # Show progress for large datasets
            if max_tweets > 1000 and batch_end % 10000 == 0:
                self.logger.info(f"📝 Generated {batch_end:,} / {max_tweets:,} tweets...")
        
        self.logger.info(f"✅ Successfully generated {len(tweets_data):,} mock tweets for query: {query}")
        return pd.DataFrame(tweets_data)

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
