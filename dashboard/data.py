import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from urllib.parse import urlencode
import plotly.graph_objects as go
import pydeck as pdk

from constants import *

@st.cache_data
def load_local_data(file_path):
    # Ensure the path is correct
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File not found: {file_path}")
        return None

@st.cache_data
def load_local_data_for_evolution_chart(file_path):
    # Ensure the path is correct
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
        return df
    else:
        st.error(f"File not found: {file_path}")
        return None

@st.cache_data
def load_data(BASE_URL):

    SELECT_COLS = ",".join([
        "id", "case_number", "date",
        "block", "iucr", "primary_type", "description", "location_description",
        "arrest", "domestic",
        "beat", "district", "ward", "community_area",
        "fbi_code",
        "year",
        "latitude", "longitude",
        "location"
    ])

    chunks = []
    offset = 0

    for page in range(MAX_PAGES):
        params = {
            "$select": SELECT_COLS,
            "$where": f"date between '{START}' and '{END}'",
            "$order": "date",
            "$limit": LIMIT,
            "$offset": offset
        }

        url = BASE_URL + "?" + urlencode(params)

        df_part = pd.read_csv(url)

        if df_part.empty:
            print(f"Stop: page {page+1} is empty. Done.")
            break

        chunks.append(df_part)
        offset += LIMIT

        print(f"Page {page+1}: rows={len(df_part)}, next_offset={offset}")
        time.sleep(SLEEP_SEC)

    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("Final shape:", df.shape)
    return df

@st.cache_data
def load_data_for_evolution_chart(BASE_URL):
    chunks = []
       
    SELECT_COLS = "date,year,latitude,longitude"
    
    for yr in range(2001, 2025):
        print(f"  Fetching data for {yr}...", end="\r")
        params = {
            "$select": SELECT_COLS,
            "$where": f"year={yr}",
            "$limit": 15000,  # 15k rows per year = ~360k rows total (Perfect size)
            "$order": "date"
        }
        url = BASE_URL + "?" + urlencode(params)
        
        try:
            chunk = pd.read_csv(url)
            chunks.append(chunk)
        except Exception as e:
            print(f"  Error fetching {yr}: {e}")

    print("\nmerging and saving...")
    df = pd.concat(chunks, ignore_index=True)

    return df

@st.cache_data
def prepare_top_districts(df, start_year=START_YEAR, end_year=2025, top_n=10):
  
    DISTRICT_MAP = {
        1.0: '1: Central (Downtown)',
        2.0: '2: Wentworth (South)',
        3.0: '3: Grand Crossing (South)',
        4.0: '4: South Chicago (South)',
        5.0: '5: Calumet (South)',
        6.0: '6: Gresham (South)',
        7.0: '7: Englewood (South)',
        8.0: '8: Chicago Lawn (Southwest)',
        9.0: '9: Deering (South)',
        10.0: '10: Ogden (West)',
        11.0: '11: Harrison (West)',
        12.0: '12: Near West (Central/West)',
        14.0: '14: Shakespeare (Northwest)',
        15.0: '15: Austin (Far West)',
        16.0: '16: Jefferson Park (Northwest)',
        17.0: '17: Albany Park (Northwest)',
        18.0: '18: Near North (Downtown)',
        19.0: '19: Town Hall (North)',
        20.0: '20: Lincoln (North)',
        22.0: '22: Morgan Park (Far South)',
        24.0: '24: Rogers Park (Far North)',
        25.0: '25: Grand Central (Northwest)'
    }

    df_dist = df[
        (df['date'].dt.year >= start_year) &
        (df['date'].dt.year <= end_year)
    ].copy()

    df_dist['district'] = pd.to_numeric(df_dist['district'], errors='coerce')

    district_counts = (
        df_dist['district']
        .value_counts()
        .nlargest(top_n)
        .reset_index()
    )

    district_counts.columns = ['District_ID', 'Count']

    district_counts['Label'] = (
        district_counts['District_ID']
        .map(DISTRICT_MAP)
        .fillna(district_counts['District_ID'].astype(str))
    )

    return district_counts
