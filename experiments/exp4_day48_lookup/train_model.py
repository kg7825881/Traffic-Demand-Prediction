import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def run_model_pipeline():
    if not os.path.exists('train_final.csv') or not os.path.exists('test_final.csv'):
        print("❌ Error: Processed data missing.")
        return

    print("📖 Loading datasets into memory...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')

    print("\n--- STEP 2: Day Distribution ---")
    print(train_df["day"].value_counts())

    print("\n--- STEP 3: Build Historical Lookup Tables (Day 48 Only) ---")
    day48_df = train_df[train_df["day"].astype(str) == '48']
    
    print("\n--- STEP 5: Verify Lookup Source ---")
    print("Lookup source rows day distribution:")
    print(day48_df["day"].value_counts())

    lookup_A = day48_df.groupby(['geo5', 'timestamp'], observed=False)['demand'].agg(['mean', 'median', 'std', 'count']).reset_index()
    lookup_A = lookup_A.rename(columns={'mean': 'hist_geo5_ts_mean', 'median': 'hist_geo5_ts_median', 'std': 'hist_geo5_ts_std', 'count': 'hist_geo5_ts_count'})

    lookup_B = day48_df.groupby(['geo4', 'timestamp'], observed=False)['demand'].agg(['mean', 'median', 'std', 'count']).reset_index()
    lookup_B = lookup_B.rename(columns={'mean': 'hist_geo4_ts_mean', 'median': 'hist_geo4_ts_median', 'std': 'hist_geo4_ts_std', 'count': 'hist_geo4_ts_count'})

    lookup_C = day48_df.groupby(['RoadType', 'timestamp'], observed=False)['demand'].agg(['mean', 'median', 'std', 'count']).reset_index()
    lookup_C = lookup_C.rename(columns={'mean': 'hist_road_ts_mean', 'median': 'hist_road_ts_median', 'std': 'hist_road_ts_std', 'count': 'hist_road_ts_count'})

    print("\n--- STEP 4: Merge Features ---")
    train_df = train_df.merge(lookup_A, on=['geo5', 'timestamp'], how='left')
    train_df = train_df.merge(lookup_B, on=['geo4', 'timestamp'], how='left')
    train_df = train_df.merge(lookup_C, on=['RoadType', 'timestamp'], how='left')

    test_df = test_df.merge(lookup_A, on=['geo5', 'timestamp'], how='left')
    test_df = test_df.merge(lookup_B, on=['geo4', 'timestamp'], how='left')
    test_df = test_df.merge(lookup_C, on=['RoadType', 'timestamp'], how='left')

    hist_cols = [
        'hist_geo5_ts_mean', 'hist_geo5_ts_median', 'hist_geo5_ts_std', 'hist_geo5_ts_count',
        'hist_geo4_ts_mean', 'hist_geo4_ts_median', 'hist_geo4_ts_std', 'hist_geo4_ts_count',
        'hist_road_ts_mean', 'hist_road_ts_median', 'hist_road_ts_std', 'hist_road_ts_count'
    ]

    for col in hist_cols:
        train_missing_pct = train_df[col].isnull().mean() * 100
        test_missing_pct = test_df[col].isnull().mean() * 100
        print(f"Missing {col}: Train={train_missing_pct:.2f}%, Test={test_missing_pct:.2f}%")
        global_mean = train_df['demand'].mean()
        train_df[col] = train_df[col].fillna(global_mean)
        test_df[col] = test_df[col].fillna(global_mean)

    train_df.drop(columns=['timestamp'], inplace=True)
    test_df.drop(columns=['timestamp'], inplace=True)

    cat_cols = [
        'RoadType',
        'LargeVehicles',
        'Landmarks',
        'Weather',
        'day',
        'road_capacity_index',
        'geohash',
        'geo4',
        'geo5',
        'geo6'
    ]
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
        if col in test_df.columns:
            test_df[col] = test_df[col].astype('category')

    print("📊 Generating frequency encoding features...")
    geo_cols = ['geohash', 'geo4', 'geo5', 'geo6']
    for col in geo_cols:
        freq_map = train_df[col].astype(str).value_counts().to_dict()
        train_df[f"{col}_freq"] = train_df[col].astype(str).map(freq_map).fillna(0).astype(float)
        test_df[f"{col}_freq"] = test_df[col].astype(str).map(freq_map).fillna(0).astype(float)

        print(f"\n{col}_freq description:")
        print(train_df[f"{col}_freq"].describe())
        print(f"Number of unseen categories in test for {col}: {test_df[col].astype(str).map(freq_map).isna().sum()}")

    target_col = 'demand'
    test_idx = test_df['Index']
    features = [c for c in train_df.columns if c not in [target_col, 'Index']]
    
    X = train_df[features].copy()
    y = train_df[target_col].copy()
    X_test = test_df[features].copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.015,
        'num_leaves': 127,
        'max_depth': -1,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42
    }

    print("train_df['geohash'].nunique():", train_df['geohash'].nunique())
    print("train_df['geo4'].nunique():", train_df['geo4'].nunique())
    print("train_df['geo5'].nunique():", train_df['geo5'].nunique())
    print("train_df['geo6'].nunique():", train_df['geo6'].nunique())

    print("🏋️ Training with Smoothed Out-of-Fold Target Encoding...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
        X_test_fold = X_test.copy()

        # ========================================================
        # REGULARIZED TARGET ENCODING WITH SMOOTHING
        # ========================================================
        encode_col = 'road_capacity_index'
        smoothing_factor = 20  # Weight given to global mean for small sample groups
        
        # Calculate group properties on training fold only
        global_mean = y_train.mean()
        group_stats = y_train.groupby(X_train[encode_col], observed=False).agg(['mean', 'count'])
        
        # Apply the explicit smoothing equation
        smoothed_vals = (
            (group_stats['mean'] * group_stats['count']) + (global_mean * smoothing_factor)
        ) / (group_stats['count'] + smoothing_factor)
        
        # Map values out securely to flat arrays
        train_mapped = X_train[encode_col].map(smoothed_vals).to_numpy(dtype=float)
        val_mapped = X_val[encode_col].map(smoothed_vals).to_numpy(dtype=float)
        test_mapped = X_test_fold[encode_col].map(smoothed_vals).to_numpy(dtype=float)

        # Fallback for any group that never appeared in training split
        train_mapped[np.isnan(train_mapped)] = global_mean
        val_mapped[np.isnan(val_mapped)] = global_mean
        test_mapped[np.isnan(test_mapped)] = global_mean

        # Inject back as an unconstrained regularized feature
        X_train = X_train.assign(smoothed_demand_avg=train_mapped)
        X_val = X_val.assign(smoothed_demand_avg=val_mapped)
        X_test_fold = X_test_fold.assign(smoothed_demand_avg=test_mapped)
        # ========================================================

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / kf.n_splits

        fold_score = max(0, 100 * r2_score(y_val, oof_preds[val_idx]))
        print(f"   🔹 Fold {fold + 1} Competition Metric Score: {fold_score:.4f}")

    final_cv_score = max(0, 100 * r2_score(y, oof_preds))
    print(f"\n🏆 Overall Cross-Validation Score with Smoothed Encoding: {final_cv_score:.4f} 🏆")

    submission = pd.DataFrame({'Index': test_idx, 'demand': test_preds})
    submission['demand'] = submission['demand'].clip(lower=0)
    
    submission.to_csv('submission_exp4_day48_lookup.csv', index=False)
    print("💾 Finished! Smoothed 'submission_exp4_day48_lookup.csv' generated cleanly.")
    
    # Export feature importance
    importance = model.feature_importance(importance_type='split')
    feat_imp = pd.DataFrame({'feature': model.feature_name(), 'importance': importance})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False)
    feat_imp.to_csv('feature_importance_exp4.csv', index=False)

if __name__ == "__main__":
    run_model_pipeline()