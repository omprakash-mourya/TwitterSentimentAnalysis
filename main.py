"""
Main entry point for Streamlit Cloud deployment
GPU-Accelerated Twitter Sentiment Analysis Dashboard
"""

# Import the main app
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from app.streamlit_app import *
