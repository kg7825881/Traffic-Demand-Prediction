# 🚦 Bengaluru Traffic Demand Prediction Engine

An advanced spatiotemporal machine learning pipeline built to optimize mobility and forecast traffic congestion surges across Bengaluru. Developed as a high-performance solution for the **Flipkart Gridlock Hackathon 2.0**, this implementation achieves a robust **93.26% local cross-validation accuracy** and a **87.62+ score on the live competition leaderboard**.


## 🌟 Key Engineering Architectural Features

* **Geographical Coordinate Decoding:** Decoded high-cardinality raw string `geohash` features into continuous numeric spatial dimensions (`latitude` and `longitude`) to allow precise tree-splitting boundaries.
* **Cyclical Temporal Mapping:** Transformed daily timestamps using circular trigonometric functions ($\sin$/$\cos$ transformations) to preserve sequential time continuity (e.g., mapping 23:59 close to 00:01).
* **Domain-Specific Interaction Modeling:** Synthesized localized indicators including contextual traffic rush windows (`is_rush_hour`), road bottleneck throughput capacities (`road_capacity_index`), and localized weather risk impact scores (`weather_severity`).
* **Skewed Target Optimization:** Implemented an out-of-fold `log1p` math transformation on the right-tailed demand surge target to stabilize variance and prevent outlier gradient blindness.

---

## 📂 Project Repository Structure

```text
├── parse_spatial.py        # Stage 1: Geohash decoding and geographical coordinate generation
├── prepare_final_data.py   # Stage 2: Temporal extraction, feature interactions, and dtype cleansing
├── train_model.py          # Stage 3: 5-Fold cross-validated, log-transformed LightGBM Regressor
├── .gitignore              # Secure rule definitions preventing raw data tracking leaks
└── README.md               # Pipeline documentation and architectural blueprint
```

## 💾 Dataset Access
The datasets used in this project are part of the Flipkart Gridlock Hackathon 2.0. 
* To run this pipeline, download `train.csv` and `test.csv` from the competition portal and place them directly into the root directory of this workspace.

## 🛠️ Execution Pipeline
To reproduce the optimal submission file locally from scratch, run the scripts sequentially within your terminal environment:

### 1. Environment Initialization
Ensure all dependencies are present within your virtual workspace environment:
```bash
pip install pandas numpy pygeohash lightgbm scikit-learn
```

### 2. Spatial Engineering Step
Decodes geographic hashes to numeric coordinate coordinates from the raw datasets:
```bash
python parse_spatial.py
```

### 3. Structural Preprocessing & Feature Synthesis
Generates cyclical features, handles missing values, and creates cross-feature combinations:
```bash
python prepare_final_data.py
```

### 4. Model Training & Evaluation
Executes the leaf-wise LightGBM training process using 5-Fold cross-validation and generates the final submission:
```bash
python train_model.py
```

## 📊 Evaluation & Validation Metrics
* Local 5-Fold Cross-Validation ($R^2$
  Space): ~93.26%
* Public Live Leaderboard Score: 87.62871

The pipeline relies on a leaf-wise asymmetric LightGBM architecture configured with a slower learning rate (`0.015`) and deep tree layouts (`num_leaves=127`) to capture volatile micro-surges across localized intersections without overfitting the baseline distributions.