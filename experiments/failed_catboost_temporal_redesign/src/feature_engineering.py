import pandas as pd
import numpy as np
from .config import Config
from .utils import timer, logger
from .spatial_features import add_spatial_features
from .historical_features import create_historical_features

def add_time_features(df):
    with timer("Time Features"):
        # Original timestamp is retained (we don't drop it)
        # Parse timestamp (format HH:MM)
        temp_dt = pd.to_datetime(df['timestamp'], format='%H:%M', errors='coerce')
        df['hour'] = temp_dt.dt.hour
        df['minute'] = temp_dt.dt.minute
        
        # Fallback if parsing fails
        if df['hour'].isnull().all():
            df['hour'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(int)
            df['minute'] = 0
            
        df['hour'] = df['hour'].fillna(0).astype(int)
        df['minute'] = df['minute'].fillna(0).astype(int)

        # Cyclic encodings
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        # Time slot (15-min intervals over 24 hours = 96 slots)
        df['time_slot'] = (df['hour'] * 4 + df['minute'] // 15).astype(str)
        
        # Peak indicators
        df['is_morning_peak'] = (((df['hour'] >= 7) & (df['hour'] <= 10))).astype(int)
        df['is_evening_peak'] = (((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
        df['is_peak'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        return df

def add_interactions(df):
    with timer("Interaction Features"):
        # Combine categoricals as strings
        df['RoadType_geo4'] = df['RoadType'].astype(str) + "_" + df['geo4'].astype(str)
        df['RoadType_geo5'] = df['RoadType'].astype(str) + "_" + df['geo5'].astype(str)
        df['RoadType_timestamp'] = df['RoadType'].astype(str) + "_" + df['timestamp'].astype(str)
        df['RoadType_Lanes'] = df['RoadType'].astype(str) + "_" + df['NumberofLanes'].astype(str)
        
        return df

def preprocess_pipeline(train, test):
    logger.info("Starting Preprocessing Pipeline")
    
    # 1. Spatial
    train, kmeans_model = add_spatial_features(train, n_clusters=Config.N_CLUSTERS, is_train=True)
    test, _ = add_spatial_features(test, n_clusters=Config.N_CLUSTERS, is_train=False, kmeans_model=kmeans_model)
    
    # 2. Time
    train = add_time_features(train)
    test = add_time_features(test)
    
    # 3. Interactions
    train = add_interactions(train)
    test = add_interactions(test)
    
    # 4. Historical Lags (Strictly from Day 48)
    train, test = create_historical_features(train, test, train_day=Config.TRAIN_DAY)
    
    # 5. Type Casting
    for col in Config.CAT_FEATURES:
        if col in train.columns:
            train[col] = train[col].fillna('Missing').astype(str).astype('category')
        if col in test.columns:
            test[col] = test[col].fillna('Missing').astype(str).astype('category')
            
    logger.info("Preprocessing complete.")
    return train, test
