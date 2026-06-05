import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

def analyze_data():
    results = {}
    
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # 1. Cardinality
    print("Calculating cardinality...")
    cardinality = {col: int(train[col].nunique()) for col in train.columns}
    results['cardinality'] = cardinality
    
    # Check geohash specifically
    if 'geohash' in train.columns:
        train['geo3'] = train['geohash'].astype(str).str[:3]
        train['geo4'] = train['geohash'].astype(str).str[:4]
        train['geo5'] = train['geohash'].astype(str).str[:5]
        train['geo6'] = train['geohash'].astype(str).str[:6]
        
        test['geo3'] = test['geohash'].astype(str).str[:3]
        test['geo4'] = test['geohash'].astype(str).str[:4]
        test['geo5'] = test['geohash'].astype(str).str[:5]
        test['geo6'] = test['geohash'].astype(str).str[:6]
        
        results['geohash_cardinality'] = {
            'geo3': int(train['geo3'].nunique()),
            'geo4': int(train['geo4'].nunique()),
            'geo5': int(train['geo5'].nunique()),
            'geo6': int(train['geo6'].nunique()),
            'geohash': int(train['geohash'].nunique())
        }

    # 4. Duplicate patterns
    print("Detecting duplicates...")
    train_dups = int(train.drop(columns=['Index']).duplicated().sum())
    test_dups = int(test.drop(columns=['Index']).duplicated().sum())
    results['duplicates'] = {'train': train_dups, 'test': test_dups}
    
    # 5. Train-test distribution drift (Adversarial Validation)
    print("Detecting drift...")
    common_cols = [c for c in train.columns if c in test.columns and c not in ['Index', 'demand']]
    adv_train = train[common_cols].copy()
    adv_test = test[common_cols].copy()
    
    adv_train['is_test'] = 0
    adv_test['is_test'] = 1
    adv_df = pd.concat([adv_train, adv_test], axis=0)
    
    # Simple label encoding for adversarial validation
    for col in adv_df.columns:
        if adv_df[col].dtype == 'object':
            adv_df[col] = adv_df[col].astype('category').cat.codes
            
    X_adv = adv_df.drop('is_test', axis=1)
    y_adv = adv_df['is_test']
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_adv.fillna(-999), y_adv)
    from sklearn.metrics import roc_auc_score
    adv_auc = roc_auc_score(y_adv, rf.predict(X_adv.fillna(-999)))
    results['adversarial_validation_auc'] = float(adv_auc)
    
    # 2 & 6. Feature Importance and Variance explained
    print("Estimating feature importance...")
    train_encoded = train.copy()
    for col in train_encoded.columns:
        if train_encoded[col].dtype == 'object':
            train_encoded[col] = train_encoded[col].astype('category').cat.codes
            
    features = [c for c in train_encoded.columns if c not in ['Index', 'demand']]
    X_imp = train_encoded[features].fillna(-999)
    y_imp = train_encoded['demand']
    
    rf_imp = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_imp.fit(X_imp, y_imp)
    
    importances = {features[i]: float(rf_imp.feature_importances_[i]) for i in range(len(features))}
    results['feature_importance'] = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # Grouped importance (Temporal vs Spatial vs Infrastructure vs Weather)
    temporal_cols = ['day', 'timestamp']
    spatial_cols = ['geohash', 'geo3', 'geo4', 'geo5', 'geo6']
    infra_cols = ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']
    weather_cols = ['Temperature', 'Weather']
    
    grouped_imp = {
        'temporal': sum([importances.get(c, 0) for c in temporal_cols]),
        'spatial': sum([importances.get(c, 0) for c in spatial_cols]),
        'infrastructure': sum([importances.get(c, 0) for c in infra_cols]),
        'weather': sum([importances.get(c, 0) for c in weather_cols])
    }
    results['grouped_importance'] = grouped_imp
    
    # 7. Geohash vs Lat/Lon
    # We need to decode geohash to get lat/lon for train
    print("Comparing geohash vs lat/lon...")
    try:
        import pygeohash as pgh
        def decode_geohash(gh):
            try:
                return pgh.decode(gh)
            except:
                return (np.nan, np.nan)
        coords = train['geohash'].apply(decode_geohash)
        train_encoded['lat'] = [c[0] for c in coords]
        train_encoded['lon'] = [c[1] for c in coords]
        
        features_latlon = [c for c in train_encoded.columns if c not in ['Index', 'demand', 'geohash', 'geo3', 'geo4', 'geo5', 'geo6']]
        rf_latlon = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf_latlon.fit(train_encoded[features_latlon].fillna(-999), y_imp)
        latlon_r2 = r2_score(y_imp, rf_latlon.predict(train_encoded[features_latlon].fillna(-999)))
        
        features_geo = [c for c in train_encoded.columns if c not in ['Index', 'demand', 'lat', 'lon']]
        rf_geo = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf_geo.fit(train_encoded[features_geo].fillna(-999), y_imp)
        geo_r2 = r2_score(y_imp, rf_geo.predict(train_encoded[features_geo].fillna(-999)))
        
        results['geohash_vs_latlon'] = {'latlon_train_r2': float(latlon_r2), 'geohash_train_r2': float(geo_r2)}
    except ImportError:
        results['geohash_vs_latlon'] = "pygeohash not installed"

    with open('forensic_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Analysis complete. Results saved to forensic_results.json")

if __name__ == "__main__":
    analyze_data()
