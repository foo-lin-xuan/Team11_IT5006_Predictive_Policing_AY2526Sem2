import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from urllib.parse import urlencode
import plotly.graph_objects as go
import pydeck as pdk

from constants import *

def chart_yearly_trend(crimes_by_year):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=crimes_by_year.index,
        y=crimes_by_year.values,
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=6),
        name="Crimes"
    ))

    fig.update_layout(
        title=f"Number of Crimes by Year ({START_YEAR} - 2025)",
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        template="plotly_white",
        yaxis=dict(
            tickformat=","  # thousands separator
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, width='stretch')

def chart_hourly_trend(crimes_by_hour):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=crimes_by_hour.index,
        y=crimes_by_hour.values,
        name="Crimes"
    ))

    fig.update_layout(
        title=f"Number of Crimes by Hour of Day ({START_YEAR} - 2025)",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Crimes"
    )

    st.plotly_chart(fig, width='stretch')

def chart_weekly_trend(crimes_by_weekday):
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
        title=f"Number of Crimes by Weekday ({START_YEAR} - 2025)",
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

def chart_heatmap(df):
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
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
    )

    st.pydeck_chart(deck)

def chart_evolution(df_hist, id, default_option=0):
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

    chart_heatmap(df_era)

def chart_top_districts(district_counts):
    x_min = district_counts["Count"].min()
    x_max = district_counts["Count"].max()
    pad = (x_max - x_min) * 0.15

    district_counts = district_counts.sort_values("Count", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=district_counts["Count"],
            y=district_counts["Label"],
            orientation="h",

            text=[f"{x:,}" for x in district_counts["Count"]],
            textposition="outside",

            marker=dict(
                color=district_counts["Count"],
                colorscale="Reds",
                showscale=False
            ),

            hovertemplate=
            "<b>%{y}</b><br>" +
            "Incidents: %{x:,}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"Top 10 High-Crime Districts by Community Area ({START_YEAR} – 2025)",
        xaxis=dict(
            title="Total Reported Incidents",
            tickformat=",",
            range=[x_min - pad, x_max + pad]
        ),
        yaxis=dict(
            title="",
            automargin=True
        ),
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_white"
    )

    st.plotly_chart(fig, width="stretch")   
