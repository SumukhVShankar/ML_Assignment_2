"""
Breast Cancer Wisconsin (Diagnostic) - Multi-Model Classification Trainer
==========================================================================

Trains five classification models on the Breast Cancer Wisconsin dataset,
evaluates six metrics for each, and persists the fitted pipelines together
with a held-out test set for use by the Streamlit application.

Dataset   : Breast Cancer Wisconsin (Diagnostic)
Source     : UCI ML Repository / sklearn.datasets.load_breast_cancer
Instances : 569   |   Features : 30 (all numeric)   |   Task : binary

Design notes
------------
* Every model is wrapped in a single sklearn Pipeline so that the exact
  preprocessing (median imputation + standard scaling) travels with the
  estimator. This makes each saved artifact fully self-contained: the
  Streamlit app only has to call `pipeline.predict(X)` on raw feature
  columns, with no risk of train/serve preprocessing skew.
* A fixed random seed keeps the train/test split and the stochastic models
  reproducible.
* The same held-out test split is written to `test_data.csv` so the grader
  can upload it to the deployed app and reproduce the reported metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
TARGET_COLUMN = "diagnosis"          # 0 = malignant, 1 = benign
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Human-friendly key -> saved filename stem
MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def build_models() -> dict[str, Pipeline]:
    """Return one Pipeline per classifier.

    Preprocessing is identical across models (impute -> scale) so the
    comparison isolates the effect of the estimator itself. SVM and Logistic
    Regression need `probability`/decision scores for a meaningful AUC, which
    the pipelines expose downstream.
    """
    preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    def make(estimator) -> Pipeline:
        return Pipeline(preprocessing + [("clf", estimator)])

    return {
        "Logistic Regression": make(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": make(
            DecisionTreeClassifier(random_state=RANDOM_STATE)
        ),
        "KNN": make(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make(GaussianNB()),
        # Wrap SVC in CalibratedClassifierCV so predict_proba is available
        # for AUC without the deprecated `probability=True` flag.
        "Support Vector Machine": make(
            CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=RANDOM_STATE),
                ensemble=False,
            )
        ),
    }


def positive_class_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Continuous scores for the positive class, used for AUC.

    Prefers calibrated probabilities; falls back to the signed distance to
    the decision boundary when a model does not expose `predict_proba`.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute the six required metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_scores = positive_class_scores(pipeline, X_test)
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_test, y_scores)), 4),
        "Precision": round(float(precision_score(y_test, y_pred)), 4),
        "Recall": round(float(recall_score(y_test, y_pred)), 4),
        "F1": round(float(f1_score(y_test, y_pred)), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    # --- Load dataset -------------------------------------------------------
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()  # 0 = malignant, 1 = benign

    print(f"Loaded dataset: {X.shape[0]} instances, {X.shape[1]} features.")

    # --- Train / test split (stratified to preserve class balance) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    # --- Train, evaluate, persist each model --------------------------------
    models = build_models()
    metrics_table: dict[str, dict[str, float]] = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        metrics_table[name] = metrics

        out_path = MODEL_DIR / f"{MODEL_FILENAMES[name]}.joblib"
        joblib.dump(pipeline, out_path)
        print(f"  [saved] {name:<24} -> {out_path.name}  |  {metrics}")

    # --- Persist feature order so the app can validate uploads --------------
    feature_metadata = {
        "feature_names": list(X.columns),
        "target_column": TARGET_COLUMN,
        "target_mapping": {"0": "malignant", "1": "benign"},
        "positive_label": 1,
    }
    joblib.dump(feature_metadata, MODEL_DIR / "feature_metadata.joblib")

    # --- Write the test set the grader will upload --------------------------
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nWrote held-out test data -> {TEST_DATA_PATH.name} ({len(test_df)} rows)")

    # --- Save metrics table (also used to render README comparison) ---------
    METRICS_PATH.write_text(json.dumps(metrics_table, indent=2), encoding="utf-8")
    print(f"Wrote metrics table      -> {METRICS_PATH.relative_to(PROJECT_ROOT)}")

    # --- Console summary ----------------------------------------------------
    summary = pd.DataFrame(metrics_table).T
    print("\n================ Model Comparison ================")
    print(summary.to_string())
    best = summary["MCC"].idxmax()
    print(f"\nBest model by MCC: {best}")


if __name__ == "__main__":
    main()
