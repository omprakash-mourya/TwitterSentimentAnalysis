import streamlit as st

st.title("🐦 Twitter Sentiment Analysis")
st.write("✅ App is working on Streamlit Cloud!")

import pandas as pd

# Simple demo
data = pd.DataFrame({
    'Sentiment': ['Positive', 'Negative', 'Neutral'],
    'Count': [45, 25, 30]
})

st.bar_chart(data.set_index('Sentiment'))
st.dataframe(data)
