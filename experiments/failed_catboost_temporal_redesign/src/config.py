import os

class Config:
    # Paths
    ROOT_DIR = "."
    DATA_DIR = ROOT_DIR
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    MODEL_DIR = os.path.join(ROOT_DIR, "models")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
    
    # Feature Engineering
    N_CLUSTERS = 20
    
    # Modeling
    SEED = 42
    TARGET = "demand"
    
    # Features lists (populated dynamically but strictly typed)
    CAT_FEATURES = [
        "geohash", "geo6", "geo5", "geo4", "geo3",
        "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Weather",
        "timestamp", "time_slot",
        "RoadType_geo4", "RoadType_geo5", "RoadType_timestamp", "RoadType_Lanes"
    ]
    
    # Validation
    # Train is day 48, Val is day 49 (from train set)
    TRAIN_DAY = 48
    VAL_DAY = 49

    @classmethod
    def setup(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
