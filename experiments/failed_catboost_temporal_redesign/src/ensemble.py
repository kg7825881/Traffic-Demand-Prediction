import pandas as pd
import numpy as np
import os
from .config import Config
from .utils import logger

def blend_predictions(cb_test, lgb_test, cb_weight=0.7, lgb_weight=0.3):
    """
    Blends the predictions using a weighted average.
    CatBoost receives higher weight due to superior native categorical handling.
    """
    logger.info(f"Blending predictions: {cb_weight*100}% CatBoost, {lgb_weight*100}% LightGBM")
    
    blended = (cb_test * cb_weight) + (lgb_test * lgb_weight)
    
    # Clip predictions at 0 since demand cannot be negative
    blended = np.clip(blended, a_min=0, a_max=None)
    
    return blended

def create_submission(test_idx, preds, filename="submission.csv"):
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    sub = pd.DataFrame({
        'Index': test_idx,
        'demand': preds
    })
    sub.to_csv(filepath, index=False)
    logger.info(f"Submission saved successfully to {filepath}")
