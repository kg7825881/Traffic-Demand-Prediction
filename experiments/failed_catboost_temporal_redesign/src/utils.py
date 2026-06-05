import pandas as pd
import time
import logging
from contextlib import contextmanager

def setup_logger(name="Gridlock2"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

@contextmanager
def timer(name):
    t0 = time.time()
    logger.info(f"[{name}] start")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.1f} s")

def load_data(train_path, test_path):
    with timer("Load Data"):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        return train, test
