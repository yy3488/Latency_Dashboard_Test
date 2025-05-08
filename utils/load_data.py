# utils/load_data.py
import pandas as pd
import numpy as np
import pycountry_convert as pc
import pycountry
import streamlit as st

def country_to_continent(code):
    try:
        return pc.country_alpha2_to_continent_code(code)
    except:
        return None

@st.cache_data
def load_data():
    data_url = "https://www.dropbox.com/scl/fi/2o01yjrep18j5e6ir56kd/p2025_04_18-p2025_04_23.csv?rlkey=lfk8lbwvuxqw8b2uhylwjhpvp&st=cw1l118o&dl=1"
    col_names = [
        'taskid', 'clientid', 'clientip', 'clientcountry', 'clientisp', 'clientlat', 'clientlng',
        'destcid', 'destip', 'destcountry', 'destisp', 'destlat', 'destlng', 'timestamp', 'ttl'
    ]
    df = pd.read_csv(data_url, header=None, names=col_names, on_bad_lines='skip')
    keep_cols = [
        'clientid', 'clientcountry', 'clientisp',
        'destcid', 'destcountry', 'destisp',
        'timestamp', 'ttl'
    ]
    df = df[keep_cols]
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
