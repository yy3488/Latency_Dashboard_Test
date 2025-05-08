import pandas as pd
from utils.load_data import country_to_continent
import streamlit as st

@st.cache_data(max_entries=5)
def compute_summary_table(df):
    df_combined = pd.DataFrame({
        'nodeid': pd.concat([df['clientid'], df['destcid']], ignore_index=True),
        'country': pd.concat([df['clientcountry'], df['destcountry']], ignore_index=True),
        'isp': pd.concat([df['clientisp'], df['destisp']], ignore_index=True)
    })

    if 'group' in df.columns:
        df_combined['group'] = pd.concat([df['group'], df['group']], ignore_index=True)

    df_combined.dropna(subset=['nodeid', 'country', 'isp'], inplace=True)

    if 'group' in df.columns:
        summary = df_combined.groupby(['group', 'country']).agg(
            unique_nodes=('nodeid', 'nunique'),
            isp_count=('isp', 'nunique')
        ).reset_index()
    else:
        summary = df_combined.groupby('country').agg(
            unique_nodes=('nodeid', 'nunique'),
            isp_count=('isp', 'nunique')
        ).reset_index()

    summary['continent'] = summary['country'].apply(country_to_continent)
    return summary
