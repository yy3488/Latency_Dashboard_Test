# pages/ 3_Country_Heatmap.py
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

st.header("🗺️ Cross-Country Latency Analysis")
st.caption("Analyze latency between countries. Filter by date range, continent, or specific countries to customize the heatmap.")

with st.expander("Country-Level Insights", expanded=True):
    date_range3 = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key="cc_date")
    selected_continent = st.selectbox("Select Continent (optional)", [None] + sorted(data['clientcontinent'].dropna().unique()), key="continent_filter")
    custom_countries = st.multiselect("Select Specific Countries (optional)", sorted(data['clientcountry'].dropna().unique()), key="custom_country")

    # Filter data by date
    filtered_cc = data.copy()
    if len(date_range3) == 2:
        filtered_cc = filtered_cc[(filtered_cc['date'] >= date_range3[0]) & (filtered_cc['date'] <= date_range3[1])]

    country_summary = compute_summary_table(filtered_cc)

    # Select countries by logic
    if selected_continent:
        top_countries = country_summary[
            country_summary['continent'] == selected_continent
            ].head(15)['country'].tolist()

    elif custom_countries:
        top_countries = custom_countries
    else:
        top_countries = (
            filtered_cc.groupby('clientcountry')['clientid'].nunique()
            .nlargest(30)
            .index.tolist()
        )

    # Filter rows where both countries are in the top list
    cc_filtered = filtered_cc[
        (filtered_cc['clientcountry'].isin(top_countries)) &
        (filtered_cc['destcountry'].isin(top_countries))
        ].copy()

    # Create symmetric pair keys
    cc1 = cc_filtered['clientcountry']
    cc2 = cc_filtered['destcountry']
    cc_filtered.loc[:, 'country_pair'] = np.where(cc1 < cc2, cc1 + '_' + cc2, cc2 + '_' + cc1)

    # Group and pivot for heatmap
    st.subheader("Country-to-Country Latency Heatmap")
    st.caption("Symmetric heatmap showing average latency between selected countries.")
    heatmap_data = cc_filtered.groupby('country_pair')['ttl'].mean().reset_index()
    heatmap_data[['country1', 'country2']] = heatmap_data['country_pair'].str.split('_', expand=True)

    pivot_table = heatmap_data.pivot(index='country1', columns='country2', values='ttl')
    pivot_symmetric = pivot_table.combine_first(pivot_table.T)

    # Plot heatmap
    fig_cc_heatmap = px.imshow(
        pivot_symmetric,
        color_continuous_scale="YlGnBu",
        title="Symmetric Country-to-Country Latency Heatmap",
        labels={"x": "Country", "y": "Country", "color": "Avg Latency (ms)"}
    )
    st.plotly_chart(fig_cc_heatmap, use_container_width=True)

    del filtered_cc, cc_filtered, heatmap_data, pivot_table, pivot_symmetric
    del cc1, cc2, fig_cc_heatmap, country_summary
    gc.collect()