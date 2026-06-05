import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os

def run_model_pipeline():
    if not os.path.exists('train_final.csv') or not os.path.exists('test_final.csv'):
        print("❌ Error: Processed data missing.")
        return

    print("📖 Loading datasets into memory...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')

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
    y = train_df[target_col].copy()
    X_test = test_df[features].copy()

    # 5-Fold Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    # Tuned hyperparameters
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

    print("🏋️ Training with Safe Out-of-Fold Target Encoding...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
        X_test_fold = X_test.copy()

        # ========================================================
        # DYNAMIC TARGET ENCODING (No Leakage)
        # ========================================================
        # Group by highly detailed traffic blocks to extract true historical means
        encode_col = 'road_capacity_index'
        
        # Calculate historical mean from training fold indices only
        train_encoded = y_train.groupby(X_train[encode_col]).mean()
        global_mean = y_train.mean() # Fallback for unseen groups

        # Map values securely
        X_train['hist_demand_avg'] = X_train[encode_col].map(train_encoded).fillna(global_mean)
        X_val['hist_demand_avg'] = X_val[encode_col].map(train_encoded).fillna(global_mean)
        X_test_fold['hist_demand_avg'] = X_test_fold[encode_col].map(train_encoded).fillna(global_mean)
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
    print(f"\n🏆 Overall Cross-Validation Score with Target Encoding: {final_cv_score:.4f} 🏆")

    submission = pd.DataFrame({'Index': test_idx, 'demand': test_preds})
    submission['demand'] = submission['demand'].clip(lower=0)
    submission.to_csv('final_submission.csv', index=False)
    print("💾 Finished! Enhanced 'final_submission.csv' generated.")

if __name__ == "__main__":
    run_model_pipeline()