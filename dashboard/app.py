import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from urllib.parse import urlencode
import plotly.graph_objects as go
import pydeck as pdk

import utils


# Change these to load data from local file for faster speed
LOAD_LOCAL_DATA = True 
LOCAL_DATA_2016_2025_FILEPATH = "../data/chicago_crime_2016_2025_raw.csv"
LOCAL_DATA_2001_2024_FILEPATH = "../data/chicago_crime_2001_2024.csv"

BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
 

st.title("Chicago Crime Data Analysis Dashboard")

# Loading Data
data_load_state = st.text('Loading large amount of data... This may take a while, please wait...')
if LOAD_LOCAL_DATA:
    df = utils.load_local_data(LOCAL_DATA_2016_2025_FILEPATH)
else:
    df = utils.load_data()
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

utils.chart_hourly_trend(crimes_by_hour)
utils.chart_weekly_trend(crimes_by_weekday)

st.markdown("**Evolution of Crime Hotspots (2001 - 2024)**")

# Loading data for evolution chart
data_load_state = st.text('Loading large amount of data... This may take a while, please wait...')
if LOAD_LOCAL_DATA:
    df_hist = utils.load_local_data_for_evolution_chart(LOCAL_DATA_2001_2024_FILEPATH)
else:
    df_hist = utils.load_data_for_evolution_chart()
data_load_state.text('Loading data...done!')

col1, col2 = st.columns(2)

with col1:
    utils.chart_evolution(df_hist, id="first",default_option=0)

with col2:
    utils.chart_evolution(df_hist, id="second",default_option=3)

utils.chart_regional_crime_trends(df)