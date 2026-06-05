import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import r2_score
from .config import Config
from .utils import timer, logger
import joblib

def train_catboost(train_df, test_df):
    logger.info("Training CatBoost Model...")
    
    # 1. Temporal Validation Split
    # Train = Day 48
    # Validation = Day 49 (from train set)
    # This prevents temporal leakage and matches the LB split logic.
    tr = train_df[train_df['day'] == Config.TRAIN_DAY].copy()
    val = train_df[train_df['day'] == Config.VAL_DAY].copy()
    
    features = [c for c in train_df.columns if c not in [Config.TARGET, 'Index', 'day']]
    cat_features = [c for c in features if c in Config.CAT_FEATURES]
    
    # CatBoost expects categorical features as strings
    for col in cat_features:
        tr[col] = tr[col].astype(str)
        val[col] = val[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
    X_train, y_train = tr[features], tr[Config.TARGET]
    X_val, y_val = val[features], val[Config.TARGET]
    X_test = test_df[features]
    
    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Identify cat feature indices
    cat_indices = [features.index(col) for col in cat_features]
    
    train_pool = cb.Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = cb.Pool(X_val, y_val, cat_features=cat_indices)
    test_pool = cb.Pool(X_test, cat_features=cat_indices)
    
    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'iterations': 4000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 5,
        'random_seed': Config.SEED,
        'task_type': 'CPU', # Change to 'GPU' if available
        'early_stopping_rounds': 100,
        'verbose': 100
    }
    
    with timer("CatBoost Fit"):
        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    val_preds = model.predict(val_pool)
    val_score = max(0, 100 * r2_score(y_val, val_preds))
    logger.info(f"CatBoost Validation Score (Day 49): {val_score:.4f}")
    
    # Feature Importance
    importances = model.get_feature_importance()
    feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False)
    feat_imp.to_csv(os.path.join(Config.OUTPUT_DIR, 'catboost_importance.csv'), index=False)
    
    # Save Model
    model_path = os.path.join(Config.MODEL_DIR, 'catboost_model.cbm')
    model.save_model(model_path)
    
    test_preds = model.predict(test_pool)
    
    # Return predictions to blend
    return test_preds, val_preds
