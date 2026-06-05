import pandas as pd
import numpy as np
import os

def clean_and_encode(input_filename, output_filename, is_train=True):
    if not os.path.exists(input_filename):
        print(f"❌ Error: Cannot find '{input_filename}'")
        return

    print(f"📖 Loading {input_filename}...")
    df = pd.read_csv(input_filename)

    # ==========================================
    # 1. TEMPORAL FEATURE EXTRACTION
    # ==========================================
    print("⏰ Processing temporal properties...")
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%H:%M', errors='coerce')
    df['hour'] = df['timestamp_dt'].dt.hour
    df['minute'] = df['timestamp_dt'].dt.minute
    df = df.drop(columns=['timestamp_dt'])
    
    if df['hour'].isnull().all():
        df['hour'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(float).fillna(0).astype(int)
        df['minute'] = 0

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # ==========================================
    # 2. TRAFFIC INTERACTION FEATURES
    # ==========================================
    print("🛠️ Engineering domain-specific traffic features...")
    
    df['is_rush_hour'] = (((df['hour'] >= 8) & (df['hour'] <= 11)) | 
                          ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
    
    # Force string construction to ensure it combines as text cleanly
    df['road_capacity_index'] = df['RoadType'].astype(str) + "_" + df['NumberofLanes'].astype(str)
    
    weather_map = {
        'Clear': 1, 'Sunny': 1,
        'Cloudy': 2, 'Haze': 2, 'Mist': 2,
        'Rain': 3, 'Drizzle': 3,
        'Heavy Rain': 4, 'Storm': 4, 'Thunderstorm': 4
    }
    df['weather_severity'] = df['Weather'].map(weather_map).fillna(2).astype(int)

    # ==========================================
    # 3. STRICT TYPE CASTING & CLEANING
    # ==========================================
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'day', 'road_capacity_index', 'geohash', 'geo4', 'geo5', 'geo6']
    num_cols = ['NumberofLanes', 'Temperature', 'latitude', 'longitude', 'hour', 'minute', 
                'hour_sin', 'hour_cos', 'is_rush_hour', 'weather_severity']
    
    print("🧹 Handling missing values and explicitly casting types...")
    # Explicit conversion loop to guarantee categorical formatting passes to CSV
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Missing').astype(str).astype('category')

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Final drop of raw non-numeric columns
    cols_to_drop = []
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    print(f"💾 Saving clean dataset to '{output_filename}'...")
    df.to_csv(output_filename, index=False)
    print(f"✅ Created {output_filename} successfully!\n")

if __name__ == "__main__":
    print("🚀 Launching Data Prep Pipeline...\n")
    clean_and_encode('train_spatial.csv', 'train_final.csv', is_train=True)
    clean_and_encode('test_spatial.csv', 'test_final.csv', is_train=False)
    print("🎉 All datasets are fully engineered and aligned successfully!")