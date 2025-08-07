"""
Streamlit entry point for cloud deployment
GPU-Accelerated Twitter Sentiment Analysis Dashboard
"""

import streamlit as st
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set page config first
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analysis - GPU Accelerated",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the main application
try:
    from app.streamlit_app import *
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("This app requires the full project structure to run properly.")
    st.stop()
