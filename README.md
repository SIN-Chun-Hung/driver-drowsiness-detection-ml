# Driver Alertness Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Platform](https://img.shields.io/badge/Jovian-Machine%20Learning-9cf)](https://jovian.ai/)
[![Dataset](https://img.shields.io/badge/Kaggle-Stay%20Alert!-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/stayalert/data)

## 1. Project Overview & Dataset

### Problem Statement
Staying alert while driving is one of the most critical factors for road safety. Driver fatigue, drowsiness, or lack of attention substantially increases the probability of traffic accidents. 

The objective of this project is to build an end-to-end binary classification pipeline using **Decision Trees** and **Random Forests** to accurately detect and predict whether a driver is in an alert state (`IsAlert = 1`) or not (`IsAlert = 0`), based on real-time sensor measurements.

### Dataset Source
* **Source**: [Kaggle - Stay Alert! The Ford Challenge](https://www.kaggle.com/c/stayalert/data)
* **Dataset Characteristics**:
  * **Sample Size**: Over **604,329 records** in the training dataset and **120,840 records** in the test dataset.
  * **Feature Dimensions**: 30 numerical input features categorized into:
    * **Physiological Indicators (`P1` – `P8`)**: Heart rate, blink rate, and other biometric responses.
    * **Environmental Metrics (`E1` – `E11`)**: Ambient conditions and external factors.
    * **Vehicular Attributes (`V1` – `V11`)**: Steering angle, acceleration, lateral deviation, and vehicle speed.
  * **Target Variable**: `IsAlert` (`1`: Alert, `0`: Not Alert).

---

## 2. Project Pipeline & Experimental Results

### End-to-End Workflow

1. **Data Acquisition & Exploration (EDA)**
   * Streamlined dataset download directly from Kaggle using the `opendatasets` library.
   * Converted raw CSV files into `pandas.DataFrame` structures.
   * Verified data hygiene: confirmed zero missing (`NaN`) values across all 30 feature columns.

2. **Data Splitting & Preprocessing**
   * **Trial-Based Splitting**: To prevent data leakage caused by temporal correlation within the same experimental run, data was partitioned by `TrialID` (~75% train / ~25% validation), yielding **462,976 training records** and **141,353 validation records**.
   * **Feature Normalization**: Applied `MinMaxScaler` across all 30 input features to standardize disparate numerical ranges to `[0, 1]`, preventing dominance by large-scale features.

3. **Model Development & Hyperparameter Tuning**
   * **Decision Tree Classifier**:
     * *Baseline*: Achieved 100% training accuracy but only 59.7% validation accuracy, indicating severe overfitting.
     * *Tuning*: Formulated dedicated evaluation loops to tune `max_depth` (1–20), `max_leaf_nodes` (10–150), and `min_samples_split` (2–20).
     * *Optimal Configuration*: `max_depth=8`, `max_leaf_nodes=20`, `min_samples_split=16`.
   * **Random Forest Classifier**:
     * *Baseline*: Default model achieved 100% training accuracy and 64.6% validation accuracy.
     * *Tuning*: Programmed custom plotting and diagnostic helper functions (`test_params`, `test_param_and_plot`) using `matplotlib` and `seaborn` to inspect overfitting curves across `n_estimators`, `max_features`, `max_depth`, `max_leaf_nodes`, `bootstrap`, `max_samples`, and `min_samples_split`.
     * *Optimal Configuration*: `n_estimators=150`, `max_features=10`, `max_depth=20`, `max_leaf_nodes=600`, `bootstrap=True`, `max_samples=0.1`, `min_samples_split=4`, `min_samples_leaf=2`.

4. **Inference on Unseen Test Data**
   * Applied the calibrated models to generate predictions (`IsAlert_DTC` and `IsAlert_RFC`) for all 120,840 rows in `fordTest.csv`.

---

### Comparative Evaluation

| Model Architecture | Train Accuracy | Validation Accuracy | Diagnosis & Key Observations |
| :--- | :---: | :---: | :--- |
| **Decision Tree (Baseline)** | 100.0% | 59.70% | Severe overfitting; unconstrained tree depth memorized noise. |
| **Decision Tree (Tuned - Best)** | **88.03%** | **71.80%** | **Best Performance; +12.1% accuracy gain after regularizing depth and leaf count.** |
| **Random Forest (Baseline)** | 100.0% | 64.63% | Overfitted due to unconstrained tree estimators. |
| **Random Forest (Tuned)** | 95.11% | 64.88% | Subsampling and feature restriction stabilized the variance. |

> **Key Takeaway**: Pruning tree complexity significantly enhanced out-of-sample generalization. The regularized Decision Tree outperformed the ensemble configuration on this dataset, achieving **~71.8% validation accuracy** while keeping model inference fast and interpretable.

---

## 3. Technical Skills & Resume Highlights

This project demonstrates core competencies in applied machine learning, data engineering, and statistical modeling:

* **High-Dimensional Data Preprocessing & Feature Engineering**
  * Handled large-scale numerical datasets (>600K records, 30+ sensor variables) using `pandas` and `numpy`.
  * Prevented data leakage across sequential observations by partitioning splits using experimental trial identifiers (`TrialID`).
  * Scaled feature distributions reliably via `MinMaxScaler` across training, validation, and test datasets.
* **Supervised Machine Learning & Ensemble Algorithms**
  * Implemented and benchmarked tree-based architectures (`DecisionTreeClassifier`, `RandomForestClassifier`) using `scikit-learn`.
  * Identified and mitigated bias-variance tradeoffs using accuracy metrics and error-rate tracking (`1 - score`).
* **Systematic Hyperparameter Optimization & Visualization**
  * Designed reusable utility functions to automate parameter exploration and grid iterations.
  * Visualized training vs. validation error curves using `matplotlib` and `seaborn` to detect overfitting inflection points.
* **Production-Ready Pipeline Structuring**
  * Structured an end-to-end machine learning project lifecycle from automated data ingestion to model deployment and evaluation on unseen test datasets.

---
