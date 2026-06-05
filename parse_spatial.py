import os
import pandas as pd
import pygeohash as pgh

def decode_geohash_safely(geohash_string):
    """
    Decodes a geohash string to (latitude, longitude).
    Handles potential missing or invalid values gracefully.
    """
    if pd.isna(geohash_string) or not isinstance(geohash_string, str):
        return None, None
    try:
        return pgh.decode(geohash_string)
    except Exception:
        return None, None

def process_dataset(filename, output_filename):
    """
    Loads a dataset, extracts coordinates, tracks progress, and saves the output.
    """
    if not os.path.exists(filename):
        print(f"❌ Error: Cannot find '{filename}' in the current workspace directory.")
        return

    print(f"📖 Loading {filename} into memory...")
    df = pd.read_csv(filename)
    print(f"   -> Loaded {len(df):,} rows.")

    print("🗺️ Decoding geohashes to precise latitudes and longitudes...")
    # Apply decoding mapping safely across the series
    coordinates = df['geohash'].apply(decode_geohash_safely)
    
    # Split the resulting tuples into two distinct continuous numeric feature columns
    df['latitude'] = [coords[0] for coords in coordinates]
    df['longitude'] = [coords[1] for coords in coordinates]

    print(f"💾 Saving processed dataset to '{output_filename}'...")
    df.to_csv(output_filename, index=False)
    print(f"✅ Successfully created '{output_filename}'!\n")
    
    # Preview the structural transformation
    print("Preview of engineered spatial coordinates:")
    print(df[['geohash', 'latitude', 'longitude']].head(), "\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("🚀 Starting Spatial Feature Engineering Pipeline...\n")
    
    # Route and parse both datasets natively
    process_dataset('train.csv', 'train_spatial.csv')
    process_dataset('test.csv', 'test_spatial.csv')
    
    print("🎉 All spatial parsing completed successfully!")