import pandas as pd
import numpy as np
import os

def engineer_temporal_features(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"❌ Error: Cannot find '{input_filename}'")
        return

    print(f"📖 Loading {input_filename}...")
    df = pd.read_csv(input_filename)

    print("⏰ Processing temporal columns...")
    
    # 1. Standardize timestamp mapping (assuming HH:MM format like '18:30')
    # If your dataset has timestamp as integer minutes or hours, this safely adapts.
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%H:%M', errors='coerce')
    
    # Extract structural components
    df['hour'] = df['timestamp_dt'].dt.hour
    df['minute'] = df['timestamp_dt'].dt.minute
    
    # Fallback handle if timestamp is a pure numeric float/int hour instead of string format
    if df['hour'].isnull().all():
        df['hour'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(float).fillna(0).astype(int)
        df['minute'] = 0

    # 2. Cyclical Time Encoding
    # Maps time onto a 24-hour circular clock representation
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # 3. Structural Cleaning
    # Drop intermediate processing datetime object
    df = df.drop(columns=['timestamp_dt'])
    
    # Keep track of structural health: check for unexpected Nulls
    if df['hour'].isnull().any():
        print("⚠️ Warning: Detected unparsed rows in timestamp column. Filled with baseline defaults.")
        df['hour'] = df['hour'].fillna(0)
        df['minute'] = df['minute'].fillna(0)

    print(f"💾 Saving to '{output_filename}'...")
    df.to_csv(output_filename, index=False)
    print(f"✅ Successfully completed temporal extraction for {output_filename}!\n")
    
    # Print sample block to verify engineering accuracy
    cols_to_show = ['timestamp', 'hour', 'minute', 'hour_sin', 'hour_cos']
    print(df[cols_to_show].head(), "\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("🚀 Launching Temporal Feature Engineering Step...\n")
    
    # Build directly on top of your fresh spatial data files
    engineer_temporal_features('train_spatial.csv', 'train_spatiotemporal.csv')
    engineer_temporal_features('test_spatial.csv', 'test_spatiotemporal.csv')
    
    print("🎉 Spatiotemporal data pipeline structural phase completed!")