import pandas as pd
from .utils import timer, logger

def create_historical_features(train, test, day_col='day', target_col='demand', train_day=48):
    """
    Computes historical demand strictly from the training day (day 48).
    This guarantees zero leakage into day 49 validation or test sets.
    """
    with timer("Historical Features"):
        # Isolate day 48 for aggregation
        hist_data = train[train[day_col] == train_day].copy()
        
        keys = [
            (['geohash', 'timestamp'], 'gh_ts'),
            (['geo5', 'timestamp'], 'geo5_ts'),
            (['RoadType', 'timestamp'], 'rt_ts')
        ]
        
        for group_cols, prefix in keys:
            # Calculate stats on Day 48
            stats = hist_data.groupby(group_cols)[target_col].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
            stats.rename(columns={
                'mean': f'{prefix}_hist_mean',
                'median': f'{prefix}_hist_median',
                'std': f'{prefix}_hist_std',
                'min': f'{prefix}_hist_min',
                'max': f'{prefix}_hist_max'
            }, inplace=True)
            
            # Merge back into Train
            train = train.merge(stats, on=group_cols, how='left')
            # Merge into Test
            test = test.merge(stats, on=group_cols, how='left')
            
            logger.info(f"Created historical features for {group_cols}")
            
        return train, test
