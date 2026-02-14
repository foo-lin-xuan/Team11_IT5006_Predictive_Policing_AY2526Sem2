# Change these to load data from local file
LOAD_LOCAL_DATA = False 
# LOCAL_DATA_2014_2025_FILEPATH = "./data/chicago_crime_2014_2025_raw.csv"
LOCAL_DATA_2016_2025_FILEPATH = "./data/chicago_crime_2016_2025_raw.csv"
LOCAL_DATA_2001_2024_FILEPATH = "./data/chicago_crime_2001_2024.csv"

BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

# START = "2014-01-01T00:00:00"
START = "2016-01-01T00:00:00"
END  = "2025-12-31T23:59:59"

LIMIT = 50000
SLEEP_SEC = 0.3
MAX_PAGES = 1000

# START_YEAR = 2014
START_YEAR = 2016