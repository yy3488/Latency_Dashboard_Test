# pages/4_ISP_Analysis.py (Optimized with Tabs and Sub-block Caching)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils.load_data import load_data
from utils.summary_utils import compute_summary_table

# Load data once (cached)
data = load_data()

st.header("📡 ISP Latency Analysis")
st.caption("Use the tabs below to switch between different ISP-focused analysis views.")

# Top-level global date filter
block4_date_range = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key="block4_date")
if len(block4_date_range) == 2:
    start, end = block4_date_range
    filtered_block4 = data[(data['date'] >= start) & (data['date'] <= end)].copy()
else:
    filtered_block4 = data.copy()

summary_block4 = compute_summary_table(filtered_block4)
top50_countries = summary_block4.nlargest(50, 'unique_nodes')['country'].tolist()
filtered_block4 = filtered_block4[
    filtered_block4['clientcountry'].isin(top50_countries) &
    filtered_block4['destcountry'].isin(top50_countries)
].copy()

@st.cache_data(max_entries=3)
def compute_block4_metrics(df):
    all_nodes = pd.concat([
        df[['clientid', 'clientcountry', 'clientisp']].rename(columns={
            'clientid': 'nodeid', 'clientcountry': 'country', 'clientisp': 'isp'
        }),
        df[['destcid', 'destcountry', 'destisp']].rename(columns={
            'destcid': 'nodeid', 'destcountry': 'country', 'destisp': 'isp'
        })
    ], ignore_index=True).dropna()

    isp_count = all_nodes.groupby('country')['isp'].nunique().reset_index(name='isp_count')
    node_per_isp = all_nodes.groupby(['country', 'isp'])['nodeid'].nunique().reset_index(name='node_count')
    total_nodes = node_per_isp.groupby('country')['node_count'].sum().reset_index(name='total_nodes')
    top3_sum = node_per_isp.sort_values(['country', 'node_count'], ascending=[True, False])\
                           .groupby('country').head(3)\
                           .groupby('country')['node_count'].sum().reset_index(name='top3_nodes')
    top3_ratio = pd.merge(total_nodes, top3_sum, on='country', how='left')
    top3_ratio['top3_ratio'] = top3_ratio['top3_nodes'] / top3_ratio['total_nodes']

    in_country = df[df['clientcountry'] == df['destcountry']]
    latency_df = in_country.groupby('clientcountry')['ttl'].mean().reset_index(name='avg_latency')
    latency_df = latency_df.rename(columns={'clientcountry': 'country'})

    merged = isp_count.merge(top3_ratio[['country', 'top3_ratio', 'total_nodes']], on='country', how='left')
    merged = merged.merge(latency_df, on='country', how='left')

    return merged

# Compute once
block4_metrics = compute_block4_metrics(filtered_block4)

# --- Tabs ---
tab = st.tabs(["📶 Country Overview", "📊 Same vs Diff ISP", "🗺️ ISP Heatmap"])

# Tab 1: Submodule 1
with tab[0]:
    selected_countries1 = st.multiselect("Select Countries (default: all top 50)", top50_countries, default=top50_countries, key="block4_sub1_countries")
    sub1_df = block4_metrics[block4_metrics['country'].isin(selected_countries1)]

    st.subheader("ISP Count by Country")
    fig_isp_count = px.bar(
        sub1_df.sort_values(by='isp_count', ascending=False),
        x='country', y='isp_count',
        title="ISP Count by Country",
        labels={"isp_count": "Number of ISPs", "country": "Country"}
    )
    st.plotly_chart(fig_isp_count, use_container_width=True)

    st.dataframe(sub1_df[['country', 'isp_count']].sort_values(by='isp_count', ascending=False), use_container_width=True)

    st.subheader("Top 3 ISP Ratio vs Latency")
    fig_top3_ratio = px.scatter(
        sub1_df, x='top3_ratio', y='avg_latency', text='country', size='total_nodes', hover_data=['total_nodes'],
        title="Top 3 ISP Node Ratio vs. In-Country Avg Latency",
        labels={"top3_ratio": "Top 3 ISP Ratio", "avg_latency": "Avg Latency (ms)"}
    )
    fig_top3_ratio.update_traces(textposition='top center')
    coeffs = np.polyfit(sub1_df['top3_ratio'].dropna(), sub1_df['avg_latency'].dropna(), 1)
    trendline = np.poly1d(coeffs)
    fig_top3_ratio.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=trendline(np.linspace(0, 1, 100)), mode='lines', name='Trendline'))
    st.plotly_chart(fig_top3_ratio, use_container_width=True)

    st.subheader("ISP Count vs Latency")
    fig_isp_latency = px.scatter(
        sub1_df, x='isp_count', y='avg_latency', text='country', size='total_nodes', hover_data=['total_nodes'],
        title="Total ISP Count vs. In-Country Avg Latency",
        labels={"isp_count": "Number of ISPs", "avg_latency": "Avg Latency (ms)"}
    )
    fig_isp_latency.update_traces(textposition='top center')
    coeffs2 = np.polyfit(sub1_df['isp_count'].dropna(), sub1_df['avg_latency'].dropna(), 1)
    fig_isp_latency.add_trace(go.Scatter(x=np.linspace(sub1_df['isp_count'].min(), sub1_df['isp_count'].max(), 100),
                                         y=np.poly1d(coeffs2)(np.linspace(sub1_df['isp_count'].min(), sub1_df['isp_count'].max(), 100)),
                                         mode='lines', name='Trendline'))
    st.plotly_chart(fig_isp_latency, use_container_width=True)
    del sub1_df, fig_isp_count, fig_top3_ratio, fig_isp_latency
    gc.collect()

# Tab 2: Submodule 2
with tab[1]:
    selected_countries2 = st.multiselect("Select Countries", top50_countries, default=top50_countries[:5], key="subblock2")
    sub2_df = filtered_block4[
        (filtered_block4['clientcountry'] == filtered_block4['destcountry']) &
        (filtered_block4['clientcountry'].isin(selected_countries2))
    ].copy()
    sub2_df['same_isp'] = np.where(sub2_df['clientisp'] == sub2_df['destisp'], 'Same ISP', 'Different ISP')
    fig_box = px.box(sub2_df, x='clientcountry', y='ttl', color='same_isp', title="Latency Distribution: Same vs Different ISP")
    st.plotly_chart(fig_box, use_container_width=True)
    del sub2_df, fig_box
    gc.collect()

# Tab 3: Submodule 3
with tab[2]:
    selected_country = st.selectbox("Select Country", sorted(top50_countries), key="subblock3")
    detail_df = filtered_block4[filtered_block4['clientcountry'] == selected_country].copy()
    isp_summary = detail_df.groupby('clientisp')['clientid'].nunique().reset_index(name='Node Count').sort_values(by='Node Count', ascending=False)
    st.dataframe(isp_summary, use_container_width=True)

    intra_df = detail_df[detail_df['clientcountry'] == detail_df['destcountry']].dropna(subset=['clientisp', 'destisp']).copy()
    intra_df['clientisp'] = intra_df['clientisp'].astype(str)
    intra_df['destisp'] = intra_df['destisp'].astype(str)
    intra_df['isp_pair'] = np.where(intra_df['clientisp'] < intra_df['destisp'],
                                    intra_df['clientisp'] + '_' + intra_df['destisp'],
                                    intra_df['destisp'] + '_' + intra_df['clientisp'])

    isp_latency = intra_df.groupby('isp_pair')['ttl'].mean().reset_index()
    isp_latency[['isp1', 'isp2']] = isp_latency['isp_pair'].str.split('_', expand=True)
    isp_matrix = isp_latency.pivot(index='isp1', columns='isp2', values='ttl')

    fig_isp_heatmap = px.imshow(isp_matrix, text_auto=True,
        title=f"Avg Latency Between ISPs in {selected_country}",
        labels={"x": "Dest ISP", "y": "Client ISP", "color": "Latency (ms)"})
    st.plotly_chart(fig_isp_heatmap, use_container_width=True)
    del detail_df, intra_df, isp_latency, isp_summary, fig_isp_heatmap
    gc.collect()
