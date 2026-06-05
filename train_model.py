import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

def run_model_pipeline():
    if not os.path.exists('train_final.csv') or not os.path.exists('test_final.csv'):
        print("❌ Error: Processed data missing.")
        return

    print("📖 Loading processed datasets...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')

    # Drop any leftover target encoding features to ensure clean feature space
    cols_to_drop = ['hist_demand_avg', 'spatiotemporal_demand_avg', 'smoothed_demand_avg']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'day', 'road_capacity_index']
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
        if col in test_df.columns:
            test_df[col] = test_df[col].astype('category')

    target_col = 'demand'
    test_idx = test_df['Index']
    features = [c for c in train_df.columns if c not in [target_col, 'Index']]
    
    X = train_df[features].copy()
    
    # ========================================================
    # LOG TRANSFORMATION OF TARGET (Normalizing Skewed Demand)
    # ========================================================
    y = np.log1p(train_df[target_col].copy()) # Computes log(x + 1) safely
    X_test = test_df[features].copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_log = np.zeros(len(train_df))
    test_preds_log = np.zeros(len(test_df))

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

    print(f"🚀 Training Log-Transformed LightGBM Core across 5 Folds...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain, num_boost_round=3500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Store log-space predictions
        oof_preds_log[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds_log += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

        # Invert predictions back to original space to compute true evaluation score
        y_val_true = np.expm1(y_val) # Computes exp(x) - 1
        val_pred_true = np.expm1(oof_preds_log[val_idx])
        
        fold_score = max(0, 100 * r2_score(y_val_true, val_pred_true))
        print(f"   🔹 Fold {fold + 1} Adjusted Score: {fold_score:.4f}")

    # ========================================================
    # INVERT GLOBAL TARGETS BACK TO REAL TRAFFIC DEMAND SPACE
    # ========================================================
    y_true_global = np.expm1(y)
    oof_preds_global = np.expm1(oof_preds_log)
    test_preds_global = np.expm1(test_preds_log)

    final_cv_score = max(0, 100 * r2_score(y_true_global, oof_preds_global))
    print(f"\n🏆 Final Log-Transformed Framework CV Score: {final_cv_score:.4f} 🏆")

    submission = pd.DataFrame({'Index': test_idx, 'demand': test_preds_global})
    submission['demand'] = submission['demand'].clip(lower=0)
    submission.to_csv('final_submission.csv', index=False)
    print("💾 Complete! Optimized submission file generated.")

if __name__ == "__main__":
    run_model_pipeline()