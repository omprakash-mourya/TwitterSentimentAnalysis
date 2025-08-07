# 🐦 Real-Time Twitter Sentiment Analysis Dashboard

A comprehensive, fully local Twitter sentiment analysis tool using Streamlit and transformer-based NLP models. Analyze sentiment for any topic without requiring Twitter API keys!

## 🚀 Features

- **🔍 Real-time Tweet Scraping**: Uses `snscrape` to fetch recent tweets without API limitations
- **🤖 Advanced NLP**: Employs HuggingFace's RoBERTa transformer model for accurate sentiment classification
- **📊 Interactive Dashboard**: Beautiful Streamlit interface with real-time visualizations
- **☁️ Word Clouds**: Generate sentiment-specific word clouds for visual text analysis
- **⚠️ Smart Alerts**: Automatic alerts for high negative sentiment topics
- **💾 Export Functionality**: Download analysis results as CSV
- **🎯 Topic Flexibility**: Analyze any keyword, hashtag, or topic

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **NLP Model**: HuggingFace Transformers (RoBERTa)
- **Data Scraping**: snscrape
- **Visualizations**: Plotly, Matplotlib, Seaborn, WordCloud
- **Data Processing**: Pandas, NumPy
- **Deep Learning**: PyTorch

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RealTimeSentimentApp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🏗️ Project Structure

```
RealTimeSentimentApp/
├── app/
│   ├── streamlit_app.py         # Main Streamlit dashboard
│   └── tweet_scraper.py         # Tweet scraping using snscrape
├── models/
│   └── sentiment_model.py       # RoBERTa sentiment analysis model
├── utils/
│   └── text_cleaning.py         # Tweet preprocessing utilities
├── data/
│   └── scraped_tweets.csv       # Saved tweet data (generated)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🎯 How to Use

1. **Start the Application**:
   - Run `streamlit run app/streamlit_app.py`
   - Open your browser to the displayed local URL

2. **Analyze Tweets**:
   - Enter any topic in the sidebar (e.g., "artificial intelligence", "#crypto", "climate change")
   - Select the number of tweets to analyze (50-500)
   - Click "Analyze Tweets"

3. **Explore Results**:
   - View sentiment distribution in pie and bar charts
   - Explore word clouds for different sentiment categories
   - Browse individual tweets with sentiment scores
   - Filter and sort tweets by various criteria

4. **Export Data**:
   - Download your analysis as CSV for further research
   - Load previous analyses for comparison

## 🔧 Key Components

### Tweet Scraper (`app/tweet_scraper.py`)
- Uses `snscrape` for robust tweet collection
- No API keys or rate limits
- Configurable search parameters
- Automatic data cleaning and validation

### Sentiment Model (`models/sentiment_model.py`)
- Leverages `cardiffnlp/twitter-roberta-base-sentiment`
- State-of-the-art transformer architecture
- Efficient batch processing
- Fallback model support

### Text Cleaning (`utils/text_cleaning.py`)
- Removes URLs, mentions, and special characters
- Preserves meaningful content
- Handles edge cases and malformed text
- Optimized for sentiment analysis

### Streamlit Dashboard (`app/streamlit_app.py`)
- Responsive, modern UI
- Real-time progress tracking
- Interactive visualizations
- Comprehensive error handling

## 📊 Analysis Features

### Sentiment Classification
- **Positive**: Optimistic, happy, supportive content
- **Negative**: Critical, angry, disappointed content  
- **Neutral**: Factual, balanced, or unclear sentiment

### Visualizations
- **Pie Chart**: Overall sentiment distribution
- **Bar Chart**: Tweet counts by sentiment
- **Word Clouds**: Most common words per sentiment
- **Data Table**: Detailed tweet analysis with metrics

### Smart Alerts
- Automatic detection of high negative sentiment (>60%)
- Real-time warnings for controversial topics
- Business intelligence insights

## 🎨 Sample Use Cases

- **Brand Monitoring**: Track public opinion about your company
- **Product Launch**: Gauge reception of new products/services
- **Event Analysis**: Monitor sentiment during live events
- **Political Analysis**: Understand public opinion on policies
- **Market Research**: Analyze consumer sentiment trends
- **Crisis Management**: Early detection of negative sentiment spikes

## 🔧 Customization

### Model Configuration
```python
# Change the sentiment model in models/sentiment_model.py
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # Default
# or
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Alternative
```

### Search Parameters
```python
# Modify search in app/tweet_scraper.py
query = f"{topic} lang:en -is:retweet"  # English tweets, no retweets
# Add filters like:
# - "min_replies:5" for engagement
# - "since:2024-01-01" for date range
# - "place:USA" for location
```

### UI Customization
- Modify colors, themes, and layouts in `streamlit_app.py`
- Add new chart types using Plotly/Matplotlib
- Customize CSS styling for branding

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app/streamlit_app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance Tips

- **Model Caching**: The sentiment model is cached for faster subsequent runs
- **Batch Processing**: Tweets are processed in batches for efficiency
- **Memory Management**: Large datasets are handled with chunking
- **Error Handling**: Robust fallbacks for network and model issues

## 🔒 Privacy & Ethics

- **No Personal Data Storage**: Only public tweet content is analyzed
- **Respectful Scraping**: Built-in delays to avoid overwhelming servers
- **No API Abuse**: Uses public interfaces responsibly
- **Transparent Analysis**: All processing steps are visible and auditable

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace** for providing excellent pre-trained models
- **Streamlit** for the amazing dashboard framework
- **snscrape** for reliable Twitter data access
- **Cardiff NLP** for the RoBERTa sentiment model

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation in each module
- Review the error messages in the Streamlit interface

---

**Built with ❤️ for the NLP and data science community**
