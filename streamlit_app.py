# Home Page: streamlit_app.py
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Global Latency Dashboard", layout="wide")
st.title("🌐 Global Latency Dashboard")
st.markdown("""
Welcome to the **Global Latency Analysis Dashboard**.

Use the sidebar to navigate to each specific analysis module:
- Global Latency Overview
- Continent Analysis
- Country-to-Country Heatmap
- ISP Latency Analysis
- Time-Based Trends
""")
