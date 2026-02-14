import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from urllib.parse import urlencode
import plotly.graph_objects as go
import pydeck as pdk

from data import *
from charts import *

# Change these to load data from local file
LOAD_LOCAL_DATA = False 
LOCAL_DATA_2014_2025_FILEPATH = "./data/chicago_crime_2014_2025_raw.csv"
LOCAL_DATA_2001_2024_FILEPATH = "./data/chicago_crime_2001_2024.csv"

BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
 

st.title("Chicago Crime Data Analysis Dashboard")

# Load Data
data_load_state = st.text('Loading large amount of data... This may take a while, please wait...')
if LOAD_LOCAL_DATA:
    df = load_local_data(LOCAL_DATA_2014_2025_FILEPATH)
else:
    df = load_data(BASE_URL)
data_load_state.text('Loading data...done!')

# Data Preprocessing
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["hour"] = df["date"].dt.hour
df["weekday"] = df["date"].dt.day_name()
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["weekday"] = pd.Categorical(df["weekday"], categories=order, ordered=True)


## Temporal Patterns
# Data Preparation (Temporal Patterns)
crimes_by_year = df.groupby("year", observed=False).size()
crimes_by_hour = df.groupby("hour", observed=False).size()
crimes_by_weekday = df.groupby("weekday", observed=False).size()

# Charts (Temporal Patterns)
chart_yearly_trend(crimes_by_year)
chart_hourly_trend(crimes_by_hour)
chart_weekly_trend(crimes_by_weekday)


## Spatial Patterns
# Data Preparation (Spatial Patterns)
df_geo = df.dropna(subset=["latitude", "longitude"])
df_sample = df_geo.sample(n=100000, random_state=42)

# Charts (Spatial Patterns)
st.markdown("**Crimes Spatial Distribution (2014 - 2025)**")
chart_heatmap(df_sample[["latitude", "longitude"]])


## Top 10 High-Crime Districts
district_counts = prepare_top_districts(df)
chart_top_districts(district_counts)


## Spatiotemporal
st.markdown("**Evolution of Crime Hotspots (2001 - 2024)**")

# Data Preparation (Spatiotemporal)
data_load_state = st.text('Loading large amount of data... This may take a while, please wait...')
if LOAD_LOCAL_DATA:
    df_hist = load_local_data_for_evolution_chart(LOCAL_DATA_2001_2024_FILEPATH)
else:
    df_hist = load_data_for_evolution_chart(BASE_URL)
data_load_state.text('Loading data...done!')

# Charts, side-by-side layout (Spatiotemporal)
col1, col2 = st.columns(2)
with col1:
    chart_evolution(df_hist, id="first",default_option=0)

with col2:
    chart_evolution(df_hist, id="second",default_option=3)
