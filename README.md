# Breast Cancer Wisconsin — Multi-Model Classification & Streamlit App

Machine Learning Assignment 2 · M.Tech (AIML) · BITS Pilani WILP

An end-to-end classification project: five ML models trained on the Breast
Cancer Wisconsin (Diagnostic) dataset, evaluated on six metrics, and served
through an interactive Streamlit app that scores an uploaded test CSV using
the pre-trained models (no retraining at inference time).

---

## a. Problem Statement

Given 30 numeric features computed from digitized images of a fine-needle
aspirate (FNA) of a breast mass, predict whether the tumor is **malignant**
or **benign**. This is a **binary classification** problem. Early, accurate
classification supports clinical decision-making, so the cost of a false
negative (missing a malignant case) is especially important — which is why we
look beyond raw accuracy to AUC, recall, and MCC.

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI ML Repository (also bundled in `sklearn.datasets.load_breast_cancer`) |
| Instances | 569 |
| Features | 30 (all numeric, continuous) |
| Target | `diagnosis` — `0 = malignant`, `1 = benign` |
| Class balance | 212 malignant / 357 benign |
| Task | Binary classification |

The 30 features are the mean, standard error, and "worst" (largest) values of
ten cell-nucleus measurements: radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, and fractal dimension.

The dataset satisfies the assignment constraints: **30 features (≥ 12)** and
**569 instances (≥ 500)**.

**Data files in the repository:**

| File | Rows | Purpose |
|---|---|---|
| `data/breast_cancer_full.csv` | 569 | The complete original dataset (all instances). |
| `data/train_data.csv` | 426 | 75% split used to fit the models. |
| `test_data.csv` | 143 | 25% held-out split — upload this one into the Streamlit app. |

All three contain the 30 feature columns plus the `diagnosis` target column.
Per the assignment ("upload only test data"), the app is designed for
`test_data.csv`; the full and train files are included for full reproducibility.

## c. GitHub Repository Link

> **https://github.com/SumukhVShankar/ML_Assignment_2**

The repository contains the complete source code, `requirements.txt`, this
`README.md`, the test data (`test_data.csv`), and the five saved models under
`model/`.

## d. Models Used

Five classifiers were trained on the **same** dataset, each wrapped in a
scikit-learn `Pipeline` with identical preprocessing (median imputation →
standard scaling) so the comparison isolates the estimator. Metrics below are
computed on the held-out test split (`test_data.csv`).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9231 | 0.9234 | 0.9540 | 0.9222 | 0.9379 | 0.8378 |
| KNN | 0.9790 | 0.9923 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Support Vector Machine | 0.9790 | 0.9969 | 0.9888 | 0.9778 | 0.9832 | 0.9553 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall. Top accuracy (0.986), AUC (0.998) and MCC (0.970). The classes are close to linearly separable after scaling, which suits a linear model — it is both the strongest and the simplest/most interpretable choice here. |
| Decision Tree | Weakest of the five (MCC 0.838, AUC 0.923). A single unpruned tree overfits the training data and gives hard, piecewise splits, so its probability estimates are poor — reflected in the much lower AUC. |
| KNN | Very strong (MCC 0.956) and notably achieves **perfect recall (1.000)** — it misses no malignant case in the test set, which is clinically valuable. Standard scaling is essential for a distance-based model like this. |
| Naive Bayes | Solid AUC (0.988) but the lowest precision (0.936). Its feature-independence assumption is violated (the "mean/se/worst" features are highly correlated), which hurts precision even though ranking (AUC) stays good. |
| Support Vector Machine | Nearly tied with KNN (MCC 0.955) and high precision (0.989). The RBF kernel captures the non-linear boundary well; calibrated probabilities give a strong AUC (0.997). A reliable, well-rounded performer. |
| **Overall, Winner for your dataset?** | **Logistic Regression** — it leads on Accuracy, AUC, F1 and MCC while being the most interpretable and lightest model. If the priority were minimizing missed malignant cases specifically, KNN (perfect recall) would be the runner-up. |

## e. Streamlit App

The app (`app.py`) provides:

- **CSV upload** of test data (upload `test_data.csv`).
- **Model selection dropdown** — pick a single model, or "Compare all models".
- **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix** and full **classification report**.

The app loads the saved pipelines from `model/` and evaluates the uploaded
data directly. **It never retrains** — every number is the saved model applied
to your test CSV.

> **Live App:** **https://2026aj05222-mlassignment2.streamlit.app/**

## f. Project Structure

```
ML_Assignment_2/
├── app.py                 # Streamlit application (inference only, no retraining)
├── train_models.py        # Trains + evaluates + saves all 5 models and test data
├── requirements.txt       # Pinned dependencies for Streamlit Cloud
├── README.md              # This file
├── test_data.csv          # Held-out test split (143 rows, 30 features + target)
├── data/
│   ├── breast_cancer_full.csv   # Full original dataset (569 rows)
│   └── train_data.csv           # Training split (426 rows)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── feature_metadata.joblib   # feature order + target info used by the app
    └── metrics.json              # cached evaluation metrics
```

---

## Reproducing the results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all five models (writes to model/ and test_data.csv)
python train_models.py

# 3. Run the app locally
streamlit run app.py
```

Then open the app, upload `test_data.csv`, and explore the models.

## Notes on Reproducibility

- A fixed `random_state = 42` governs the train/test split and all stochastic
  models, so results are deterministic.
- Preprocessing is embedded in each saved pipeline, so there is no train/serve
  skew: the app feeds raw feature columns straight into `pipeline.predict`.
- `scikit-learn` is pinned in `requirements.txt` because the serialized models
  are version-sensitive.
