# pages/1_Global_Latency.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import pycountry
from itertools import combinations
import plotly.graph_objects as go
import gc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils.load_data import load_data
from utils.summary_utils import compute_summary_table

data = load_data()

st.header("🌐 Global Latency Overview")
st.caption("Analyze global latency trends and node activity over time. Adjust the date range and aggregation frequency to explore the data.")

with st.expander("1. Global Metrics and Trends", expanded=True):
    # Filters: Aggregation Frequency + Date Range
    freq = st.selectbox("Aggregation Frequency", options=["Daily", "Weekly", "Monthly"])
    overview_date_range = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key='overview_date')

    df1 = data.copy()
    if len(overview_date_range) == 2:
        start, end = overview_date_range
        df1 = df1[(df1['date'] >= start) & (df1['date'] <= end)]

    # Grouping logic based on frequency
    if freq == "Weekly":
        df1['group'] = df1['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    elif freq == "Monthly":
        df1['group'] = df1['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)
    else:
        df1['group'] = df1['date']

    # 1. Number of Unique Nodes Over Time
    st.subheader("Unique Nodes Over Time")
    st.caption("Trend of unique nodes based on the selected aggregation frequency.")
    summary_table = compute_summary_table(df1)
    unique_nodes_trend = summary_table.groupby('group')['unique_nodes'].sum().reset_index()
    fig_nodes = px.line(unique_nodes_trend, x='group', y='unique_nodes', title="Unique Nodes Over Time", labels={'group': freq, 'unique_nodes': 'Number of Unique Nodes'})
    st.plotly_chart(fig_nodes, use_container_width=True)

    # 2. Summary Stats
    st.subheader("Summary Statistics")
    st.caption("Key latency metrics and node counts for the selected date range.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Latency (ms)", f"{df1['ttl'].mean():,.2f}")
    col2.metric("Max Latency (ms)", f"{df1['ttl'].max():,.2f}")
    col3.metric("Min Latency (ms)", f"{df1['ttl'].min():,.2f}")
    col4.metric("Number of Nodes", f"{df1['clientid'].nunique():,}")

    # 3. Average Latency Over Time
    st.subheader("Average Latency Over Time")
    st.caption("Trend of average latency based on the selected aggregation frequency.")
    latency_grouped = df1.groupby('group')['ttl'].mean().reset_index()
    fig_latency = px.line(latency_grouped, x='group', y='ttl', title="Average Latency Over Time", labels={'ttl': 'Avg Latency (ms)', 'group': freq})
    st.plotly_chart(fig_latency, use_container_width=True)

    del df1, summary_table, unique_nodes_trend, latency_grouped, fig_nodes, fig_latency
    gc.collect()