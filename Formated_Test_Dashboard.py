import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import pycountry
from itertools import combinations
import plotly.graph_objects as go

# ---------- Page Setup ----------
st.set_page_config(page_title="Global Latency Analysis", layout="wide")

# ---------- Helper Functions ----------
def iso2_to_iso3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None

import pycountry_convert as pc
def country_to_continent(code):
    try:
        return pc.country_alpha2_to_continent_code(code)
    except:
        return None

# ---------- Load Data ----------
@st.cache_data
def load_data():
    data_url = "https://www.dropbox.com/scl/fi/2o01yjrep18j5e6ir56kd/p2025_04_18-p2025_04_23.csv?rlkey=lfk8lbwvuxqw8b2uhylwjhpvp&st=cw1l118o&dl=1"
    col_names = [
        'taskid', 'clientid', 'clientip', 'clientcountry', 'clientisp', 'clientlat', 'clientlng',
        'destcid', 'destip', 'destcountry', 'destisp', 'destlat', 'destlng', 'timestamp', 'ttl'
    ]
    df = pd.read_csv(data_url, header=None, names=col_names, on_bad_lines='skip')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df = df.dropna(subset=['timestamp', 'clientcountry', 'clientisp', 'destcountry', 'destisp'])
    df = df[(df['timestamp'] >= pd.Timestamp('2025-04-18')) & (df['timestamp'] <= pd.Timestamp('2025-04-23'))]
    df = df[df['ttl'] <= 100000]
    df['ttl'] = df['ttl'].replace(-1, np.nan)
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['clientcontinent'] = df['clientcountry'].apply(country_to_continent)
    df['destcontinent'] = df['destcountry'].apply(country_to_continent)
    df = df.dropna(subset=['clientcontinent', 'destcontinent'])
    return df
data = load_data()  

# ---------- Utility: Compute Unique Node & ISP Summary by Country (with or without group) ----------
@st.cache_data
def compute_summary_table(df):
    has_group = 'group' in df.columns

    client_df = df[['clientid', 'clientcountry', 'clientisp']].copy()
    client_df = client_df.rename(columns={'clientid': 'nodeid', 'clientcountry': 'country', 'clientisp': 'isp'})
    if has_group:
        client_df['group'] = df['group']

    dest_df = df[['destcid', 'destcountry', 'destisp']].copy()
    dest_df = dest_df.rename(columns={'destcid': 'nodeid', 'destcountry': 'country', 'destisp': 'isp'})
    if has_group:
        dest_df['group'] = df['group']

    combined_df = pd.concat([client_df, dest_df], ignore_index=True)
    drop_cols = ['nodeid', 'country']
    if has_group:
        drop_cols.append('group')
    combined_df = combined_df.dropna(subset=drop_cols)

    if has_group:
        summary = combined_df.groupby(['group', 'country']).agg(
            unique_nodes=('nodeid', 'nunique'),
            isp_count=('isp', 'nunique')
        ).reset_index()
    else:
        summary = combined_df.groupby('country').agg(
            unique_nodes=('nodeid', 'nunique'),
            isp_count=('isp', 'nunique')
        ).reset_index()

    summary['continent'] = summary['country'].apply(country_to_continent)

    return summary

# ---------- BLOCK 1: Global Latency Overview ----------
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

# ---------- BLOCK 2: Continent Latency Analysis ----------
st.header("🌍 Continent Latency Analysis")
st.caption("Explore latency within and across continents. Use the date range filter to narrow down the analysis period.")

with st.expander("2. Continent-Level Insights", expanded=True):
    # Filter: Date Range
    continent_date_range = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key='continent_date')
    df2 = data.copy()
    if len(continent_date_range) == 2:
        start, end = continent_date_range
        df2 = df2[(df2['date'] >= start) & (df2['date'] <= end)]

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
    df2['continent_pair'] = df2.apply(
        lambda x: '_'.join(sorted([x['clientcontinent'], x['destcontinent']])), axis=1
    )

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

# ---------- BLOCK 3: Cross-Country Latency Analysis ----------
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
    cc_filtered['country_pair'] = cc_filtered.apply(
        lambda x: '_'.join(sorted([x['clientcountry'], x['destcountry']])), axis=1
    )

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

# -------------- Block 4: ISP Latency Analysis --------------
st.header("📡 ISP Latency Analysis")
st.caption("All analysis below is restricted to the top 50 countries by unique node count based on the selected date range.")

# Global filter: Date range
block4_date_range = st.date_input("Select Date Range", [data['date'].min(), data['date'].max()], key="block4_date")
filtered_block4 = data.copy()
if len(block4_date_range) == 2:
    start, end = block4_date_range
    filtered_block4 = filtered_block4[(filtered_block4['date'] >= start) & (filtered_block4['date'] <= end)]
# Filter only Top 50 Countries
summary_block4 = compute_summary_table(filtered_block4)
top50_countries = summary_block4.nlargest(50, 'unique_nodes')['country'].tolist()
filtered_block4 = filtered_block4[
    filtered_block4['clientcountry'].isin(top50_countries) &
    filtered_block4['destcountry'].isin(top50_countries)
]

# Calculate within-Country Metrics
@st.cache_data
def compute_block4_metrics(df):
    client_df = df[['clientid', 'clientcountry', 'clientisp']].rename(
        columns={'clientid': 'nodeid', 'clientcountry': 'country', 'clientisp': 'isp'}
    )
    dest_df = df[['destcid', 'destcountry', 'destisp']].rename(
        columns={'destcid': 'nodeid', 'destcountry': 'country', 'destisp': 'isp'}
    )
    all_nodes = pd.concat([client_df, dest_df], ignore_index=True).dropna()

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

block4_metrics = compute_block4_metrics(filtered_block4)

# ---------- Block 4 - Submodule 1: Country ISP and Latency Overview ----------
with st.expander("1. Country ISP and Latency Overview", expanded=True):
    st.markdown("#### 📶 ISP Count, Concentration & Latency per Country")
    st.caption("Filtered to top 50 countries by unique node count based on selected date range.")

    # 子模块 1 的 filter：可多选国家，默认 top50 全选
    selected_countries1 = st.multiselect(
        "Select Countries (default: all top 50)",
        options=top50_countries,
        default=top50_countries,
        key="block4_sub1_countries"
    )

    # 根据筛选国家过滤 block4_metrics
    sub1_df = block4_metrics[block4_metrics['country'].isin(selected_countries1)]

    # --- 图 1：每个国家的 ISP 数量柱状图 ---
    st.subheader("ISP Count by Country")
    st.caption("Bar chart showing the number of ISPs in each selected country.")
    fig_isp_count = px.bar(
        sub1_df.sort_values(by='isp_count', ascending=False),
        x='country', y='isp_count',
        title="ISP Count by Country",
        labels={"isp_count": "Number of ISPs", "country": "Country"}
    )
    st.plotly_chart(fig_isp_count, use_container_width=True)

    st.dataframe(
        sub1_df[['country', 'isp_count']].sort_values(by='isp_count', ascending=False),
        use_container_width=True
    )

    # --- 图 2：Top3 ISP 占比 vs 平均延迟 ---
    st.subheader("Top 3 ISP Ratio vs Latency")
    st.caption("Scatter plot showing the relationship between Top 3 ISP concentration and average latency. Point size reflects total nodes.")
    # Create scatter plot without trendline
    fig_top3_ratio = px.scatter(
        sub1_df,
        x='top3_ratio',
        y='avg_latency',
        text='country',
        size='total_nodes',
        hover_data=['total_nodes'],
        title="Top 3 ISP Node Ratio vs. In-Country Avg Latency",
        labels={"top3_ratio": "Top 3 ISP Ratio", "avg_latency": "Avg Latency (ms)"}
    )

    fig_top3_ratio.update_traces(
        textposition='top center'  # Move text labels above the points
    )

    # Calculate OLS trendline manually
    x = sub1_df['top3_ratio'].dropna()
    y = sub1_df['avg_latency'].dropna()
    coeffs = np.polyfit(x, y, 1)  # Linear fit (degree=1)
    trendline = np.poly1d(coeffs)
    x_trend = np.linspace(x.min(), x.max(), 100)
    y_trend = trendline(x_trend)

    # Add trendline as a new trace
    fig_top3_ratio.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Trendline',
            line=dict(color='red')
        )
    )

    st.plotly_chart(fig_top3_ratio, use_container_width=True)

    # --- 图 3：ISP 总数量 vs 平均延迟 ---
    st.subheader("ISP Count vs Latency")
    st.caption("Scatter plot showing the relationship between the number of ISPs and average latency. Point size reflects total nodes.")
    # Create scatter plot without trendline
    fig_isp_latency = px.scatter(
        sub1_df,
        x='isp_count',
        y='avg_latency',
        text='country',
        size='total_nodes',
        hover_data=['total_nodes'],
        title="Total ISP Count vs. In-Country Avg Latency",
        labels={"isp_count": "Number of ISPs", "avg_latency": "Avg Latency (ms)"}
    )
    fig_isp_latency.update_traces(
        textposition='top center'  # Move text labels above the points
    )

    # Calculate OLS trendline manually
    x = sub1_df['isp_count'].dropna()
    y = sub1_df['avg_latency'].dropna()
    coeffs = np.polyfit(x, y, 1)  # Linear fit (degree=1)
    trendline = np.poly1d(coeffs)
    x_trend = np.linspace(x.min(), x.max(), 100)
    y_trend = trendline(x_trend)

    # Add trendline as a new trace
    fig_isp_latency.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Trendline',
            line=dict(color='red')
        )
    )
    st.plotly_chart(fig_isp_latency, use_container_width=True)

# ---------- Sub-block 2: Same vs Different ISP Latency Comparison ----------
with st.expander("2. Same vs Different ISP Latency Comparison", expanded=True):
    st.subheader("Same vs Different ISP Latency")
    st.caption("Box plot comparing latency distributions for connections within the same ISP vs different ISPs.")
    default_countries = summary_block4.nlargest(5, 'unique_nodes')['country'].tolist()
    selected_countries2 = st.multiselect("Select Countries", top50_countries, default=default_countries, key="subblock2")
    sub2_df = filtered_block4[
        (filtered_block4['clientcountry'] == filtered_block4['destcountry']) &
        (filtered_block4['clientcountry'].isin(selected_countries2))
    ].copy()

    sub2_df['same_isp'] = np.where(sub2_df['clientisp'] == sub2_df['destisp'], 'Same ISP', 'Different ISP')
    fig_box = px.box(sub2_df, x='clientcountry', y='ttl', color='same_isp',
                     title="Latency Distribution: Same vs Different ISP")
    st.plotly_chart(fig_box)

# ---------- Sub-block 3: Country ISP Detail View ----------
with st.expander("3. Detailed ISP Latency View by Country", expanded=True):
    st.subheader("ISP Details by Country")
    st.caption("Select a country to view ISP node counts and a heatmap of latency between ISPs.")
    selected_country = st.selectbox("Select Country", sorted(top50_countries), key="subblock3")
    detail_df = filtered_block4[filtered_block4['clientcountry'] == selected_country]

    isp_summary = detail_df.groupby('clientisp')['clientid'].nunique().reset_index(name='Node Count').sort_values(by='Node Count', ascending=False)
    st.dataframe(isp_summary, use_container_width=True)

    intra_df = detail_df[detail_df['clientcountry'] == detail_df['destcountry']].copy()
    intra_df = intra_df.dropna(subset=['clientisp', 'destisp'])
    intra_df['clientisp'] = intra_df['clientisp'].astype(str)
    intra_df['destisp'] = intra_df['destisp'].astype(str)
    intra_df['isp_pair'] = intra_df.apply(lambda x: '_'.join(sorted([x['clientisp'], x['destisp']])), axis=1)

    isp_latency = intra_df.groupby('isp_pair')['ttl'].mean().reset_index()
    isp_latency[['isp1', 'isp2']] = isp_latency['isp_pair'].str.split('_', expand=True)
    isp_matrix = isp_latency.pivot(index='isp1', columns='isp2', values='ttl')

    fig_isp_heatmap = px.imshow(
        isp_matrix, text_auto=True, title=f"Avg Latency Between ISPs in {selected_country}",
        labels={"x": "Dest ISP", "y": "Client ISP", "color": "Latency (ms)"}
    )
    st.plotly_chart(fig_isp_heatmap, use_container_width=True)

# -------------- Block 5: Time Analysis --------------
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

# ---------- End of Dashboard ----------
