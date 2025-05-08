# pages/2_Continent_Analysis.py
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

st.header("🌍 Continent Latency Analysis")
st.caption("Explore latency within and across continents. Use the date range filter to narrow down the analysis period.")

with st.expander("2. Continent-Level Insights", expanded=True):
    # Filter: Date Range
    continent_date_range = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key='continent_date')

    if len(continent_date_range) == 2:
        start, end = continent_date_range
        df2 = data[(data['date'] >= start) & (data['date'] <= end)].copy()  # 如果后续修改列，才加 copy
    else:
        df2 = data

    # 1. Inside the Continent
    st.subheader("Latency Within Continents")
    st.caption("Average latency for connections within the same continent.")
    same_continent = df2[df2['clientcontinent'] == df2['destcontinent']]
    continent_latency = same_continent.groupby('clientcontinent')['ttl'].mean().reset_index()

    continent_names = {
        'AF': 'Africa', 'AS': 'Asia', 'EU': 'Europe', 'NA': 'North America', 'OC': 'Oceania', 'SA': 'South America'
    }
    continent_latency['continent'] = continent_latency['clientcontinent'].map(continent_names)

    fig_continent = px.bar(
        continent_latency,
        x='continent', y='ttl',
        title="Average Latency Within Continents",
        labels={'ttl': 'Avg Latency (ms)', 'continent': 'Continent'}
    )
    st.plotly_chart(fig_continent, use_container_width=True)
    st.dataframe(continent_latency.rename(columns={'ttl': 'Avg Latency (ms)'}))

    # 2. Cross-Continent Heatmap
    st.subheader("Cross-Continent Latency")
    st.caption("Heatmap showing average latency between continents.")

    cc1 = df2['clientcontinent']
    cc2 = df2['destcontinent']
    df2.loc[:, 'continent_pair'] = np.where(cc1 < cc2, cc1 + '_' + cc2, cc2 + '_' + cc1)

    # Group and average latency per continent-pair
    heat_data = df2.groupby('continent_pair')['ttl'].mean().reset_index()
    heat_data[['cc1', 'cc2']] = heat_data['continent_pair'].str.split('_', expand=True)

    heatmap_df = heat_data.pivot(index='cc1', columns='cc2', values='ttl')
    heatmap_symmetric = heatmap_df.combine_first(heatmap_df.T)

    fig_heat = px.imshow(
        heatmap_symmetric,
        color_continuous_scale="Viridis",
        title="Symmetric Cross-Continent Latency Heatmap",
        labels={"x": "Continent", "y": "Continent", "color": "Avg Latency (ms)"}
    )
    st.plotly_chart(fig_heat, use_container_width=True)


    del df2, same_continent, continent_latency, fig_continent
    del cc1, cc2, heat_data, heatmap_df, heatmap_symmetric, fig_heat
    gc.collect()