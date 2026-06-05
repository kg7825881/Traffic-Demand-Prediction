import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os

def run_model_pipeline():
    # 1. Verify existence of final files
    if not os.path.exists('train_final.csv') or not os.path.exists('test_final.csv'):
        print("Error: Ready data missing. Run previous preprocessing blocks first.")
        return

    print("📖 Loading processed datasets...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')

    # Convert object columns to category datatype for LightGBM explicit processing
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'day']
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
        if col in test_df.columns:
            test_df[col] = test_df[col].astype('category')

    # Separate target value and drop structural columns that shouldn't enter training
    target_col = 'demand'
    
    # Store test indices safely for final mapping
    test_idx = test_df['Index']
    
    # Define features to explicitly train on
    features = [c for c in train_df.columns if c not in [target_col, 'Index']]
    
    X = train_df[features]
    y = train_df[target_col]
    X_test = test_df[features]

    print(f"📊 Training Matrix Size: {X.shape[0]:,} samples with {X.shape[1]} features.")
    print(f"🚀 Features entering the model: {features}\n")

    # 2. Setup 5-Fold Cross-Validation Framework
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    # Baseline regression parameters adjusted for spatiotemporal targets
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

    print("🏋️ Training model across 5 separate cross-validation folds...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Convert to LightGBM data structure
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train with early stopping to prevent over-fitting
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Record validation predictions
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Accumulate test prediction share
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

        # Compute metric performance per specific fold
        fold_r2 = r2_score(y_val, oof_preds[val_idx])
        fold_score = max(0, 100 * fold_r2)
        print(f"   🔹 Fold {fold + 1} Competition Metric Score: {fold_score:.4f}")

    # 3. Calculate Global Performance Metric
    global_r2 = r2_score(y, oof_preds)
    final_cv_score = max(0, 100 * global_r2)
    print(f"\n🏆 Overall Cross-Validation Score: {final_cv_score:.4f} 🏆")

    # 4. Export submission to standard structure format
    submission = pd.DataFrame({
        'Index': test_idx,
        'demand': test_preds
    })
    
    # Enforce non-negative demand rule baseline restriction
    submission['demand'] = submission['demand'].clip(lower=0)
    
    submission_file = 'final_submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"💾 File exported cleanly! Ready for leaderboard submission: '{submission_file}'")

if __name__ == "__main__":
    run_model_pipeline()