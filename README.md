# Traffic Demand Prediction – Flipkart Gridlock 2.0

## Overview

This repository contains the solution for predicting traffic demand in the Flipkart Gridlock 2.0 challenge.

We are predicting traffic demand using:
* geohash
* road metadata
* weather
* timestamp information

**Evaluation metric:** R²

---

## Repository Structure

* `data/`: Contains raw datasets (`train.csv`, `test.csv`, `sample_submission.csv`)
* `experiments/`: Isolated environments preserving every major modeling hypothesis tested.
* `submissions/`: Output directory for generated `.csv` files ready for Kaggle/competitions.
* `docs/`: Documentation and historical experiment logs.
* `parse_spatial.py`: Spatial engineering and coordinate decoding.
* `prepare_final_data.py`: Final feature alignments and transformations.
* `train_model.py`: LightGBM model training, validation, and inference.

---

## Pipeline Overview

1. **Spatial parsing**: Decodes geohashes into explicit coordinates and hierarchy levels.
2. **Temporal parsing**: (If applicable) Extracts cyclical and datetime-based features.
3. **Feature engineering**: Generates frequency encodings and spatial density features.
4. **LightGBM training**: Employs KFold CV and target encoding to train regression trees.
5. **Submission generation**: Aggregates Out-of-Fold (OOF) predictions and outputs final test targets.

---

## Experiment History

A comprehensive record of all hypotheses tested can be found here:
[`docs/experiment_log.md`](docs/experiment_log.md)

---

## Current Best Solution

**Leaderboard Score:** 91.19004

**Features:**
* geohash hierarchy
* frequency encoding
* spatial density features
* LightGBM

---

## Reproducing Results

To run the pipeline and reproduce the best solution, execute the following commands in order:

```bash
python parse_spatial.py
python parse_temporal.py
python prepare_final_data.py
python train_model.py
```

---

## Future Experiments

* spatial clustering
* CatBoost comparison
* ensemble methods
