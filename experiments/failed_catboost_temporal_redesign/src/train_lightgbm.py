import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score
from .config import Config
from .utils import timer, logger

def train_lightgbm(train_df, test_df):
    logger.info("Training LightGBM Model...")
    
    tr = train_df[train_df['day'] == Config.TRAIN_DAY].copy()
    val = train_df[train_df['day'] == Config.VAL_DAY].copy()
    
    features = [c for c in train_df.columns if c not in [Config.TARGET, 'Index', 'day']]
    cat_features = [c for c in features if c in Config.CAT_FEATURES]
    
    # LGBM requires categorical columns to be properly typed as 'category'
    for col in cat_features:
        # Convert to category type explicitly
        tr[col] = tr[col].astype('category')
        val[col] = val[col].astype('category')
        test_df[col] = test_df[col].astype('category')
        
    X_train, y_train = tr[features], tr[Config.TARGET]
    X_val, y_val = val[features], val[Config.TARGET]
    X_test = test_df[features]
    
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_features)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': Config.SEED
    }
    
    with timer("LightGBM Fit"):
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                       lgb.log_evaluation(100)]
        )
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_score = max(0, 100 * r2_score(y_val, val_preds))
    logger.info(f"LightGBM Validation Score (Day 49): {val_score:.4f}")
    
    # Save Model
    model_path = os.path.join(Config.MODEL_DIR, 'lightgbm_model.txt')
    model.save_model(model_path)
    
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    
    return test_preds, val_preds
