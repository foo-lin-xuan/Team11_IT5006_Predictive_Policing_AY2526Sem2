import json
import requests
import random
import pandas as pd
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"
STATS_FILE = "./api/models/feature_stats.json"
METADATA_FILE = "./api/models/model_metadata.json"
# NUM_SIMULATIONS = 50  # Start small to test
# PRIMARY_TYPE = "BATTERY" # High-frequency type
LOG_FILE = "simulation_logs.json"

with open(STATS_FILE, "r") as f:
    stats = json.load(f)

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

def get_random_in_range(feature_name):
    # We pull from 25% to 75% to stay realistic
    # Or use 'min' and 'max' if you want to push boundaries
    low = stats[feature_name]["25%"]
    high = stats[feature_name]["75%"]
    return random.uniform(low, high)

results_list = []
n_found = 0

# print(f"🚀 Starting simulation for {NUM_SIMULATIONS} iterations...")
print(f"🚀 Starting simulation...")

# for i in range(NUM_SIMULATIONS):
while n_found < 3:
    # 1. Generate Random Time (Focusing on night: 10 PM - 4 AM)
    # random_hour = random.choice([22, 23, 0, 1, 2, 3])
    random_hour = random.randint(0, 23)
    random_dt = datetime(2026, 4, 16, random_hour, random.randint(0, 59))
    
    # 2. Build Payload
    # Note: Using 'int()' for counts to avoid the 422 API error
    payload = {
        "date": random_dt.isoformat(),
        "primary_type": random.choice(metadata["primary_type_classes"]),
        "latitude": 41.8781, # Centered on Chicago
        "longitude": -87.6298,
        "d1_count": int(get_random_in_range("d1_count")),
        "d7_count": int(get_random_in_range("d7_count")),
        "d7_avg": get_random_in_range("d7_avg"),
        "d7_std": get_random_in_range("d7_std"),
        "arrest_count": get_random_in_range("arrest_count"),
        "domestic_count": get_random_in_range("domestic_count"),
        "d30_avg": get_random_in_range("d30_avg"),
        "d30_std": get_random_in_range("d30_std")
    }

    # 3. Call API
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()

            log_entry = {
                "input": payload,
                "predictions": {
                    "logistic_regression": {"pred": data["logistic_regression_prediction"], "prob": data["logistic_regression_probability"]},
                    "random_forest": {"pred": data["random_forest_prediction"], "prob": data["random_forest_probability"]},
                    "xgboost": {"pred": data["xgboost_prediction"], "prob": data["xgboost_probability"]},
                    "ensemble": {"pred": data["ensemble_prediction"], "prob": data["ensemble_probability"]}
                }
            }
            
            # Print a marker if we find a High Risk!
            if data["xgboost_prediction"] == 1:
                n_found += 1
                results_list.append(log_entry)
                # print(f"🎯 Found High Risk (Iter {i}): Prob {data['xgboost_probability']:.2%}")
                print(f"🎯 Found High Risk: Prob {data['xgboost_probability']:.2%}")
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Connection Error: {e}")

with open(LOG_FILE, "w") as f:
    json.dump(results_list, f, indent=2)

print(f"\n✅ Done! Saved {len(results_list)} results to {LOG_FILE}")