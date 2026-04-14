import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Geocoder

API_URL = "https://foo-lin-xuan-crime-risk-prediction.hf.space"

PRIMARY_TYPE_CLASSES = [
    "OFFENSE INVOLVING CHILDREN", "OTHER OFFENSE", "MOTOR VEHICLE THEFT",
    "CRIMINAL DAMAGE", "CRIMINAL SEXUAL ASSAULT", "DECEPTIVE PRACTICE",
    "THEFT", "BATTERY", "SEX OFFENSE", "ASSAULT", "CRIM SEXUAL ASSAULT",
    "BURGLARY", "WEAPONS VIOLATION", "CRIMINAL TRESPASS", "NARCOTICS",
    "ROBBERY", "LIQUOR LAW VIOLATION", "HOMICIDE", "PUBLIC PEACE VIOLATION",
    "INTERFERENCE WITH PUBLIC OFFICER", "STALKING", "INTIMIDATION",
    "ARSON", "HUMAN TRAFFICKING", "GAMBLING", "KIDNAPPING", "PROSTITUTION",
    "OBSCENITY", "CONCEALED CARRY LICENSE VIOLATION", "NON-CRIMINAL",
    "PUBLIC INDECENCY", "OTHER NARCOTIC VIOLATION", "RITUALISM"
]
PRIMARY_TYPE_CLASSES.sort()

# --- Page Config ---
st.set_page_config(
    page_title="Crime Risk Prediction", 
    layout="centered",
    initial_sidebar_state="collapsed"
    )

# --- Initialize Session State for Coordinates ---
# This ensures both the map and the number boxes stay in sync
if "lat" not in st.session_state:
    st.session_state.lat = 41.8781
if "lng" not in st.session_state:
    st.session_state.lng = -87.6298
if "api_results" not in st.session_state:
    st.session_state.api_results = []

st.title("🚨 Crime Risk Prediction Dashboard")
st.markdown("Enter the target details and historical context to forecast crime risk.")

tab1, tab2 = st.tabs(["🔍 Prediction Dashboard", "📏 Model Performance"])

with tab1:
    # --- SECTION 1: LOCATION SELECTION ---
    st.subheader("📍 Location")
    col_map, col_inputs = st.columns([2, 1])

    with col_map:
        # Create map centered on the current state
        m = folium.Map(location=[st.session_state.lat, st.session_state.lng], zoom_start=13)

        folium.Marker(
            [st.session_state.lat, st.session_state.lng],
            tooltip=f"Lat: {st.session_state.lat:.4f}, Lng: {st.session_state.lng:.4f}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        Geocoder(add_marker=True).add_to(m)
        # m.add_child(folium.LatLngPopup())
        
        # Render map
        map_data = st_folium(m, height=400, width="stretch", key="chicago_map")

        # --- THE SYNC LOGIC ---
        # Check if the map was clicked
        if map_data.get("last_clicked"):
            new_lat = map_data["last_clicked"]["lat"]
            new_lng = map_data["last_clicked"]["lng"]
            
            # Only update if the values are different to prevent infinite rerun loops
            if new_lat != st.session_state.lat or new_lng != st.session_state.lng:
                st.session_state.lat = new_lat
                st.session_state.lng = new_lng
                # IMPORTANT: Manually update the number_input keys so they refresh
                st.session_state.lat_box = new_lat
                st.session_state.lng_box = new_lng
                st.rerun()

    with col_inputs:
        st.write("**Coordinates**")
        
        # Define functions to handle manual typing updates
        def update_from_boxes():
            st.session_state.lat = st.session_state.lat_box
            st.session_state.lng = st.session_state.lng_box

        # Number inputs tied to the 'lat_box' and 'lng_box' session state keys
        st.number_input(
            "Latitude", 
            value=st.session_state.lat, 
            format="%.6f",
            step=0.1, 
            key="lat_box", 
        )
        
        st.number_input(
            "Longitude", 
            value=st.session_state.lng, 
            format="%.6f",
            step=0.1, 
            key="lng_box", 
        )
        
        # Sync logic: Update the master lat/lng if the boxes changed
        if st.session_state.lat_box != st.session_state.lat or st.session_state.lng_box != st.session_state.lng:
            st.session_state.lat = st.session_state.lat_box
            st.session_state.lng = st.session_state.lng_box
            st.rerun()

        st.info("Click the map to update coordinates, or type them in manually.")
    
    
    # --- Input Form ---
    with st.form("prediction_form"):
        
        # --- SECTION 2: PREDICTION TARGET DETAILS ---
        st.subheader("🎯 Details")
        
        date_input = st.date_input("Target Date to Predict", value=datetime.now(), key="pred_date")
        time_input = st.time_input("Target Time to Predict", value=datetime.now().time(), key="pred_time")
        primary_type = st.selectbox("Primary Type", PRIMARY_TYPE_CLASSES)
        
        st.divider()

        # --- SECTION 3: HISTORICAL CONTEXT & STATS ---
        st.subheader("📊 Historical Data & Statistics")
        
        # Row 1: 
        st.caption("Short-term Lags")
        col3, col4 = st.columns(2)
        with col3:
            d1_count = st.number_input("Crimes Count: 1 Day Prior", value=15)
        with col4:
            d7_count = st.number_input("Crimes Count: 7 Days Prior", value=15)

        # Row 2: 
        st.caption("7-Day Aggregates")
        col5, col6 = st.columns(2)
        with col5:
            d7_avg = st.number_input("7-Day Crimes (Avg) ", value=14.50)
            arrest_count = st.number_input("Total Arrests (Past 7 Days)", value=15)
        with col6:
            d7_std = st.number_input("7-Day Crimes (Std Dev)", value=3.84)
            domestic_count = st.number_input("Domestic-Related Crimes (Past 7 Days)", value=18)

        # Row 3: 
        st.caption("Long-term Aggregates")
        col7, col8 = st.columns(2)
        with col7:
            d30_avg = st.number_input("30-Day Crimes (Avg)", value=14.55)
        with col8:
            d30_std = st.number_input("30-Day Crimes (Std Dev)", value=4.09)

        submit_button = st.form_submit_button(label="Generate Risk Forecast", width="stretch")

    # --- API Integration ---
    if submit_button:
        dt_combined = datetime.combine(date_input, time_input).isoformat()
        
        payload = {
            "date": dt_combined,
            "primary_type": primary_type,
            "latitude": st.session_state.lat,
            "longitude": st.session_state.lng,
            "d1_count": int(d1_count),
            "d7_count": int(d7_count),
            "d7_avg": d7_avg,
            "d7_std": d7_std,
            "arrest_count": int(arrest_count),
            "domestic_count": int(domestic_count),
            "d30_avg": d30_avg,
            "d30_std": d30_std
        }

        with st.spinner("Analyzing risk patterns..."):
            try:
                response = requests.post(API_URL + "/predict", json=payload)
                response.raise_for_status()
                result = response.json()

                # Add Priority for Actionability
                def get_priority(prob):
                    # Adjusted based on XGBoost probabilities distribution.
                    # 75th percentile: 0.15360009670257568, 90th percentile: 0.9952225685119629
                    if prob > 0.90: return "❗ CRITICAL"
                    if prob > 0.50: return "🔴 HIGH"
                    if prob > 0.20: return "🟡 MED"
                    return "🟢 LOW"
                
                enriched_result = {
                    "input": payload,
                    "predictions": result,
                    "priority": get_priority(result["xgboost_probability"])
                }

                st.session_state.api_results.append(enriched_result)
                
                st.success("Analysis Complete")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {e}")
    
    if st.session_state.api_results:
        payload = st.session_state.api_results[-1]["input"]
        result = st.session_state.api_results[-1]["predictions"]

        # 1. MODEL PREDICTIONS IN COLUMNS
        st.write("#### Model Probabilities")
        m1, m2, m3, m4 = st.columns(4)
        
        models = [
            ("Logistic Regression", 'logistic_regression'),
            ("Random Forest", 'random_forest'),
            ("XGBoost (Final)", 'xgboost'),
            ("Ensemble (RF + XGB)", 'ensemble')
        ]

        cols = [m1, m2, m3, m4]

        for col, (label, key) in zip(cols, models):
            with col:
                prob = result[f'{key}_probability']
                pred = result[f'{key}_prediction']
                
                # Logic for Delta: 
                # Down Arrow, Green Color for 0 (Low), 
                # Up Arrow, Red Color for 1 (High)
                delta_val = "High Crime" if pred == 1 else "Low Crime"
                
                st.metric(
                    label=label, 
                    value=f"{prob:.2%}", 
                    delta=delta_val,
                    delta_arrow=f"{'up' if pred == 1 else 'down'}",
                    delta_color=f"{'red' if pred == 1 else 'green'}"
                )

        st.divider()

        # 2. THE VERDICT
        verdict = result.get("verdict", "N/A")
        st.markdown(f"### Final Verdict: **{verdict}**")
        st.progress(result['xgboost_probability'])
        st.caption(f"Crime Risk: {result['xgboost_probability']:.2%}")

        # --- THE RESULT MAP ---
        # Create a map centered on the selected location
        m_result = folium.Map(location=[st.session_state.lat, st.session_state.lng], zoom_start=14)
        
        # for point in st.session_state.api_results:
        color = "red" if result['xgboost_prediction'] == 1 else "green"
        icon = "exclamation-triangle" if result['xgboost_prediction'] == 1 else "check"

        folium.Marker(
            [payload['latitude'], payload['longitude']],
            popup=f"<b>{payload['primary_type']}</b><br>Risk: {result['xgboost_probability']:.1%}",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m_result)

        # Render the Result Map
        st_folium(m_result, height=300, width="stretch", key="result_map")

        st.divider()
        st.subheader("📜 Prediction History")

        if st.session_state.api_results:
            # 1. Convert to DataFrame
            df = pd.json_normalize(st.session_state.api_results)

            # 2. Create the Strategic View
            history_view = pd.DataFrame({
                "ID": df.get("predictions.request_id", "N/A"),
                "Time": pd.to_datetime(df["input.date"]).dt.strftime('%b %d, %H:%M'),
                "Location": df.apply(lambda x: f"{x['input.latitude']:.4f}, {x['input.longitude']:.4f}", axis=1),
                "Type": df["input.primary_type"],
                "Risk Score": df["predictions.xgboost_probability"].apply(lambda x: f"{x:.1%}"),
                "Verdict": df["predictions.verdict"],
                "Priority": df["priority"]
            })

            # 3. Display (Newest first)
            st.dataframe(
                history_view.iloc[::-1], 
                width="stretch",
                hide_index=True
            )
            
            col1, col2 = st.columns([1, 1])

            with col1:
                # Export Full Audit Log
                # Prepare the CSV
                csv_data = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Export Full Audit Log (CSV)",
                    data=csv_data,
                    file_name=f"crime_audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

            with col2:                 
                # Add a button to clear history
                if st.button("Clear History"):
                    st.session_state.api_results = []
                    st.rerun()


        else:
            st.info("No predictions generated yet.")

        st.divider()

        # 3. JSON IN A HIDDEN DRAWER
        with st.expander("🔍 View Raw API Response (JSON)"):
            st.json(result)
            st.write(f"**Request ID:** {result.get('request_id')}")
            st.write(f"**Processed at:** {result.get('timestamp')}")
        

@st.cache_data
def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model-info")
        response.raise_for_status()
        return response.json()
    except:
        return None

model_info = get_model_info()

with tab2:
    st.header("📏 Model Performance Metrics")
    
    if model_info:

        # Convert Metrics JSON to a pretty DataFrame
        metrics_dict = model_info['metrics']
        df_metrics = pd.DataFrame(metrics_dict).T # Transpose to get models as rows
        df_metrics = df_metrics.rename(index={'ensemble': 'ensemble (rf + xgb)'})
        
        # Format for better readability
        st.subheader("Comparison Table")
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color='lightgreen'),
            width="stretch"
        )

        # Highlight the Winner
        st.info(f"**Insight:** XGBoost leads with an F1-Score of {metrics_dict['xgboost']['f1']:.4f}")

    else:
        st.error("Could not load model metrics from API.")

# --- SIDEBAR ---
with st.sidebar:
    
    # 1. URL to API
    st.subheader("API Access")
    st.code(API_URL, language="text")
    st.link_button("Open API Docs", f"{API_URL}/docs")
    
    st.divider()
    
    # 2. Health Check
    st.subheader("System Health")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        health = response.json()

        if health.get("status") == "healthy":
            st.success("Overall Status: Healthy")
        else:
            st.warning(f"Status: {health.get('status', 'Unknown')}")

        # Display individual pipeline statuses
        with st.expander("Pipeline Details", expanded=True):
            st.write(f"**LR Pipeline:** {health.get('logistic_regression_pipeline', 'N/A')}")
            st.write(f"**RF Pipeline:** {health.get('random_forest_pipeline', 'N/A')}")
            st.write(f"**XGB Pipeline:** {health.get('xgboost_pipeline', 'N/A')}")
            st.write(f"**Total Predictions:** {health.get('total_predictions', 0)}")

    except requests.exceptions.RequestException as e:
        st.error("API Connection Error")
        st.caption("Could not reach the health check endpoint.")