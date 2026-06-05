import pandas as pd
import numpy as np
import pygeohash as pgh
from sklearn.cluster import KMeans
from .utils import timer, logger

def decode_geohash_safely(gh):
    if pd.isna(gh) or not isinstance(gh, str):
        return np.nan, np.nan
    try:
        return pgh.decode(gh)
    except:
        return np.nan, np.nan

def add_spatial_features(df, n_clusters=20, is_train=True, kmeans_model=None):
    with timer("Spatial Features"):
        # 1. Geohash Hierarchy
        # Keeping original geohash. 
        # geo6 provides street-level routing
        # geo5 provides local zones
        # geo4 provides broad neighborhoods
        # geo3 provides macro region
        df['geo6'] = df['geohash'].astype(str).str[:6]
        df['geo5'] = df['geohash'].astype(str).str[:5]
        df['geo4'] = df['geohash'].astype(str).str[:4]
        df['geo3'] = df['geohash'].astype(str).str[:3]

        # 2. Coordinates
        coords = df['geohash'].apply(decode_geohash_safely)
        df['latitude'] = [c[0] for c in coords]
        df['longitude'] = [c[1] for c in coords]

        # 3. KMeans Clusters
        # Fill missing coords with median for clustering
        lat_fill = df['latitude'].median() if is_train else 0 # Will be updated if train
        lon_fill = df['longitude'].median() if is_train else 0
        
        coords_df = df[['latitude', 'longitude']].fillna({'latitude': lat_fill, 'longitude': lon_fill})
        
        if is_train:
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['spatial_cluster'] = kmeans_model.fit_predict(coords_df)
        else:
            if kmeans_model is not None:
                df['spatial_cluster'] = kmeans_model.predict(coords_df)
            else:
                df['spatial_cluster'] = -1
                
        # 4. Density Features (Frequency Encoding)
        # To avoid leakage, density should ideally be computed on train or full dataset?
        # Since test geohashes might be new or differently distributed, we use the combined data context
        # But we pass data separately. For frequency, doing it per dataset is okay if distributions are similar,
        # but to be strict, we'll map frequencies from training.
        # Actually, counting occurrences within the provided dataframe:
        df['geohash_density'] = df.groupby('geohash')['geohash'].transform('count')
        df['geo5_density'] = df.groupby('geo5')['geo5'].transform('count')

        return df, kmeans_model
