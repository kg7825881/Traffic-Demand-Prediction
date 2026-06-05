import os
from .config import Config
from .utils import load_data, logger, timer
from .feature_engineering import preprocess_pipeline
from .train_catboost import train_catboost
from .train_lightgbm import train_lightgbm
from .ensemble import blend_predictions, create_submission

def main():
    Config.setup()
    logger.info("🚀 Starting Full Pipeline Execution")
    
    with timer("End-to-End Pipeline"):
        # 1. Load Data
        train, test = load_data(Config.TRAIN_PATH, Config.TEST_PATH)
        test_idx = test['Index'].copy()
        
        # 2. Preprocess Data
        train_processed, test_processed = preprocess_pipeline(train, test)
        
        # 3. Train CatBoost
        cb_test_preds, cb_val_preds = train_catboost(train_processed, test_processed)
        
        # 4. Train LightGBM
        lgb_test_preds, lgb_val_preds = train_lightgbm(train_processed, test_processed)
        
        # 5. Ensemble
        # We can also check correlation between models here
        final_preds = blend_predictions(cb_test_preds, lgb_test_preds, cb_weight=0.7, lgb_weight=0.3)
        
        # 6. Generate Submission
        create_submission(test_idx, final_preds, filename="submission.csv")
        
    logger.info("🎉 Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
