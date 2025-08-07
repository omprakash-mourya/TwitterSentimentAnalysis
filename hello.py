import streamlit as st

st.title("🐦 Twitter Sentiment Analysis")
st.write("Hello from Streamlit Cloud!")
st.success("✅ Basic app is working!")

# Simple demo
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'sentiment': ['positive', 'negative', 'neutral'],
    'count': [50, 30, 20]
})

st.bar_chart(data.set_index('sentiment'))
