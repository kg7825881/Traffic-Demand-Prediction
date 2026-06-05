import pandas as pd
import os

def clean_and_encode(input_filename, output_filename, is_train=True):
    if not os.path.exists(input_filename):
        print(f"❌ Error: Cannot find '{input_filename}'")
        return

    print(f"📖 Loading {input_filename}...")
    df = pd.read_csv(input_filename)

    # 1. Identify Categorical & Numeric columns based on description
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'day']
    num_cols = ['NumberOfLanes', 'Temperature', 'latitude', 'longitude', 'hour', 'minute', 'hour_sin', 'hour_cos']
    
    print("🧹 Handling missing values and casting types...")
    # Fill categorical missing values with a baseline placeholder string
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Missing').astype(str).astype('category')

    # Fill numerical missing values with median (computed safely per column)
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2. Select final features for the model
    # Drop columns we have fully engineered away or don't need for learning
    cols_to_drop = ['geohash', 'timestamp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    print(f"💾 Saving clean dataset to '{output_filename}'...")
    df.to_csv(output_filename, index=False)
    print(f"✅ Created {output_filename} successfully!\n")

if __name__ == "__main__":
    print("🚀 Launching Final Data Preparation Step...\n")
    
    # Process both datasets based on our spatiotemporal files
    clean_and_encode('train_spatiotemporal.csv', 'train_final.csv', is_train=True)
    clean_and_encode('test_spatiotemporal.csv', 'test_final.csv', is_train=False)
    
    print("🎉 All datasets are fully engineered, cleaned, and ready for modeling!")