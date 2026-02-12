import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from urllib.parse import urlencode
import plotly.graph_objects as go
import pydeck as pdk


BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

START = "2016-01-01T00:00:00"
END  = "2025-12-31T23:59:59"

LIMIT = 50000
SLEEP_SEC = 0.3
MAX_PAGES = 1000

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

@st.cache_data
def load_data():
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
def load_data_for_evolution_chart():
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

st.title("Chicago Crime Data Analysis Dashboard")

# Loading Data
data_load_state = st.text('Loading large amount of data... This may take a while, please wait...')
df = load_data()
data_load_state.text('Loading data...done!')

# Data Preprocessing
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["hour"] = df["date"].dt.hour
df["weekday"] = df["date"].dt.day_name()
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["weekday"] = pd.Categorical(df["weekday"], categories=order, ordered=True)

crimes_by_hour = df.groupby("hour", observed=False).size()
crimes_by_weekday = df.groupby("weekday", observed=False).size()

def chart_hourly_trend():
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=crimes_by_hour.index,
        y=crimes_by_hour.values,
        name="Crimes"
    ))

    fig.update_layout(
        title="Number of Crimes by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Crimes"
    )

    st.plotly_chart(fig, width='stretch')

def chart_weekly_trend():
    y_min = crimes_by_weekday.min()
    y_max = crimes_by_weekday.max()
    pad = (y_max - y_min) * 0.15

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=crimes_by_weekday.index,
        y=crimes_by_weekday.values,
        width=0.75,
        name="Crimes"
    ))

    fig.update_layout(
        title="Number of Crimes by Weekday",
        xaxis_title="Weekday",
        yaxis_title="Number of Crimes",
        xaxis=dict(tickangle=45),
        yaxis=dict(
            range=[y_min - pad, y_max + pad],
            tickformat=",",   # thousands separator
        ),
        margin=dict(l=40, r=40, t=60, b=80)
    )

    st.plotly_chart(fig, width="stretch")

def chart_evolution(id, default_option=0):
    # Era selector 
    era_mapping = {
        "1. Early 2000s (2001–2006)": (2001, 2006),
        "2. Post-Recession (2007–2012)": (2007, 2012),
        "3. Recent Past (2013–2018)": (2013, 2018),
        "4. Modern Era (2019–2024)": (2019, 2024),
    }
    selected_era = st.selectbox("Select Era", options=list(era_mapping.keys()), index=default_option, key=id)

    start_year, end_year = era_mapping[selected_era]

    df_era = df_hist[
        (df_hist['year'] >= start_year) &
        (df_hist['year'] <= end_year)
    ]

    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_era,
        get_position='[longitude, latitude]',
        radiusPixels=60,
        intensity=1,
        threshold=0.05,
        opacity=0.7
    )

    view_state = pdk.ViewState(
        latitude=41.88,
        longitude=-87.63,
        zoom=9,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        description="Evolution of Crime Hotspots (2001–2024)"
    )

    st.pydeck_chart(deck)    

def chart_regional_crime_trends():
    # Aggregate
    crime_trends = df.groupby(['year', 'district']).size().reset_index(name='incidents')

    target_districts = [1.0, 11.0, 16.0]

    df_trend = crime_trends[crime_trends['district'].isin(target_districts)]

    district_names = {
        1.0: 'Central (Downtown)',
        11.0: 'West Side (Harrison)',
        16.0: 'Northwest (Safe Zone)'
    }

    df_trend['District_Name'] = df_trend['district'].map(district_names)

    # Color mapping
    color_map = {
        'Central (Downtown)': 'red',
        'West Side (Harrison)': 'blue',
        'Northwest (Safe Zone)': 'green'
    }

    fig = go.Figure()

    # Add one line per district
    for district in df_trend['District_Name'].unique():
        df_d = df_trend[df_trend['District_Name'] == district]
        
        fig.add_trace(go.Scatter(
            x=df_d['year'],
            y=df_d['incidents'],
            mode='lines+markers',
            name=district,
            line=dict(width=3, color=color_map[district]),
            marker=dict(size=6)
        ))

    # Add COVID-19 highlight
    fig.add_vrect(
        x0=2020, 
        x1=2022, 
        fillcolor="gray", 
        opacity=0.1, 
        line_width=0, 
        annotation_text="COVID-19 Affected Era", 
        annotation_position="top left"
    )

    fig.update_layout(
        title="Diverging Destinies: Crime Trends by District Type (2016-2025)",
        xaxis_title="Year",
        yaxis_title="Total Annual Incidents",
        template="plotly_white",
        legend_title="District Archetype",
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")    

chart_hourly_trend()
chart_weekly_trend()

st.markdown("**Evolution of Crime Hotspots (2001 - 2024)**")

# Loading data for evolution chart
data_load_state = st.text('Loading large amount of data... This may take a while, please wait...')
df_hist = load_data_for_evolution_chart()
data_load_state.text('Loading data...done!')

col1, col2 = st.columns(2)

with col1:
    chart_evolution(id="first",default_option=0)

with col2:
    chart_evolution(id="second",default_option=3)

chart_regional_crime_trends()