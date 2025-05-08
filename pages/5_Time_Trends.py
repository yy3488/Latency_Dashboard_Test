# pages/ 5_Time_Trends.py
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

st.header("🕒 Time-Based Latency Analysis")
st.caption("Examine latency patterns over time, including hourly and weekday/weekend trends. Filter by date range, country, and ISP.")

with st.expander("Time-Level Insights", expanded=True):
    time_date = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key="time_date")
    time_country = st.multiselect("Select Countries", sorted(set(data['clientcountry'])), key="time_country")
    time_isp = st.multiselect("Select ISPs", sorted(set(data['clientisp'])), key="time_isp")

    time_df = data.copy()
    if len(time_date) == 2:
        time_df = time_df[(time_df['date'] >= time_date[0]) & (time_df['date'] <= time_date[1])]
    if time_country:
        time_df = time_df[time_df['clientcountry'].isin(time_country)]
    if time_isp:
        time_df = time_df[time_df['clientisp'].isin(time_isp)]

    # Hourly latency
    st.subheader("Latency Variation by Hour")
    st.caption("Line chart showing average latency for in-country connections by hour of the day.")
    time_same = time_df[time_df['clientcountry'] == time_df['destcountry']]
    time_hour = time_same.groupby('hour')['ttl'].mean().reset_index()
    fig_time_hour = px.line(time_hour, x='hour', y='ttl', title="Average Latency by Hour")
    st.plotly_chart(fig_time_hour, use_container_width=True)

    # Weekday vs Weekend
    st.subheader("Weekday vs Weekend Latency")
    st.caption("Bar chart comparing average latency on weekdays vs weekends.")
    time_df['day_type'] = np.where(time_df['weekday'] < 5, 'Weekday', 'Weekend')
    week_stat = time_df.groupby('day_type')['ttl'].mean().reset_index()
    fig_week = px.bar(week_stat, x='day_type', y='ttl', title="Latency on Weekdays vs Weekends")
    st.plotly_chart(fig_week, use_container_width=True)

    del time_df, time_same, time_hour, week_stat, fig_time_hour, fig_week
    gc.collect()