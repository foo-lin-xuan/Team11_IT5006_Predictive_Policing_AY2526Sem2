import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.tseries.holiday import USFederalHolidayCalendar

class CrimeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.PROPERTY = ['THEFT','BURGLARY','MOTOR VEHICLE THEFT','ROBBERY','ARSON',
            'CRIMINAL DAMAGE','DECEPTIVE PRACTICE']
        self.cal = USFederalHolidayCalendar()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure Date is datetime
        date_col = 'date'
        dt = pd.to_datetime(X[date_col])

        # 1. Temporal Features
        X['hour'] = dt.dt.hour
        X['day_of_week'] = dt.dt.dayofweek
        X['month'] = dt.dt.month
        X['day_of_month'] = dt.dt.day
        X['quarter'] = dt.dt.quarter
        X['year'] = dt.dt.year
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
        X['is_night'] = X['hour'].isin(list(range(0, 6)) + list(range(21, 24))).astype(int)
        X['time_period'] = pd.cut(X['hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False).astype(int)
        X['is_payday'] = X['day_of_month'].isin([1, 15]).astype(int)

        # 2. Cyclical encodings
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)

        # 3. Holiday Logic
        start_date = dt.min().normalize() - pd.Timedelta(days=7)
        end_date = dt.max().normalize() + pd.Timedelta(days=7)
        us_holidays = self.cal.holidays(start=start_date, end=end_date)
        
        # Juneteenth observed logic
        unique_years = dt.dt.year.unique()
        juneteenth_obs = []

        for y in unique_years:
            if y >= 2021:
                d = pd.Timestamp(year=y, month=6, day=19)
                if d.dayofweek == 5: d -= pd.Timedelta(days=1)
                elif d.dayofweek == 6: d += pd.Timedelta(days=1)
                juneteenth_obs.append(d)
        
        holidays = pd.to_datetime(sorted(set(us_holidays.tolist() + juneteenth_obs)))
        date_norm = dt.dt.normalize()
        
        X['is_holiday'] = date_norm.isin(holidays).astype(int)
        X['is_pre_holiday'] = (date_norm + pd.Timedelta(days=1)).isin(holidays).astype(int)
        X['is_post_holiday'] = (date_norm - pd.Timedelta(days=1)).isin(holidays).astype(int)
        X['is_long_weekend'] = ((X['is_weekend'] == 1) | (X['is_holiday'] == 1) | 
                                (X['is_pre_holiday'] == 1) | (X['is_post_holiday'] == 1)).astype(int)

        # 4. Crime Classification
        if 'primary_type' in X.columns:
            X['is_property'] = X['primary_type'].isin(self.PROPERTY).astype(int)

        # 5. Crime Trend
        X['crime_trend'] = X['d7_avg'] / X['d30_avg'].clip(lower=0.1)

        # Rename columns to align with pipelines
        column_mapping = {
            'primary_type': 'primary_type_topk',
            'latitude': 'lat_mean',
            'longitude': 'lon_mean',
            'd1_count': 'lag_1d',
            'd7_count': 'lag_7d',
            'd7_avg': 'rolling_7d',
            'd7_std': 'rolling_std_7d',
            'd30_std': 'rolling_std_30d'
        }
        X.rename(columns=column_mapping, inplace=True)

        # Drop intermediate/raw columns that shouldn't go to the model
        cols_to_drop = [date_col, 'hour', 'year'] 
        return X.drop(columns=[c for c in cols_to_drop if c in X.columns])