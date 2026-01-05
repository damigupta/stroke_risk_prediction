# Stroke Risk Prediction

A machine learning project that analyzes patient health data to predict stroke risk using feature engineering, clustering, and classification models.

## Project Overview

This project explores a dataset of ~15,000 patient records containing symptoms and health indicators to:
- Engineer meaningful risk indices from raw symptom data
- Segment patients into risk profiles using clustering
- Build and compare classification models for stroke risk prediction

## Dataset

The dataset (`strokeX.csv`) contains patient-level health indicators:

| Feature | Description |
|---------|-------------|
| `age` | Patient age (years) |
| `gender` | Male/Female |
| 15 symptom indicators | Binary (0/1): chest_pain, high_bp, irregular_heartbeat, short_breath, fatigue, dizziness, swelling, neckjaw_pain, excess_sweating, persistent_cough, nausea_vomiting, chest_discomfort, cold_extremities, sleep_apnea, anxiety |
| `stroke_risk_pct` | Estimated stroke risk (0-100%) |
| `at_risk` | Binary target variable (1 = at risk, 0 = not at risk) |

## Feature Engineering

Three custom risk indices were developed:

| Index | Formula | Purpose |
|-------|---------|---------|
| **ANRI** | stroke_risk_pct / age | Identifies patients with unusually high risk for their age |
| **SBI** | Sum of all symptoms | Quantifies total symptom burden |
| **CHRI** | 0.4×high_bp + 0.4×irregular_heartbeat + 0.2×ANRI | Composite cardiovascular risk score |

**Key Finding:** SBI shows strong correlation (r=0.68) with stroke risk. Patients with 7+ symptoms have 99.3% average risk.

## Clustering Analysis

K-Means clustering (k=3) segmented patients into distinct risk profiles:

- **Low Risk:** Younger patients, fewer symptoms
- **Medium Risk:** Middle-aged, moderate symptom burden
- **High Risk:** Older patients, high symptom count

GMM clustering validated these groupings with ARI > 0.8.

## Model Performance

Seven classification models were evaluated:

| Model | AUC | F1 Score | Accuracy |
|-------|-----|----------|----------|
| **CatBoost** | **1.000** | **0.993** | **0.991** |
| HistGradientBoosting | 0.999 | 0.992 | 0.990 |
| XGBoost | 0.999 | 0.989 | 0.986 |
| Gradient Boosting | 0.998 | 0.982 | 0.978 |
| Logistic Regression | 0.998 | 0.980 | 0.975 |
| Random Forest | 0.997 | 0.977 | 0.972 |
| KNN | 0.950 | 0.891 | 0.870 |

**Best Model:** CatBoost achieved near-perfect classification.

## Top Predictive Features

1. SBI (Symptom Burden Index)
2. Age
3. Total symptoms
4. Sleep apnea, chest pain, high BP

## Requirements
Install project dependencies. Run
```
pip install -r requirements.txt
```

Alternatively, the notebook includes `%pip install -r requirements.txt` in the first cell. By default, it is commented,
but you can uncomment it to import the dependencies when you run the notebook.

Key dependencies: pandas, numpy, scikit-learn, catboost, xgboost, matplotlib, seaborn

## Usage

Open and run `stroke_risk.ipynb` in Jupyter Notebook or JupyterLab.

## Conclusions

1. **Symptom count is the strongest predictor** - Patients with 7+ symptoms have near-certain stroke risk
2. **Age amplifies risk** - Older patients with multiple symptoms face significantly higher risk
3. **Gradient boosting models excel** - CatBoost outperformed traditional ML approaches
4. **Clustering validates clinical intuition** - Risk stratification aligns with low/medium/high groupings
