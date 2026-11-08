"""
Breast Cancer Wisconsin - Multi-Model Classifier Explorer
=========================================================

Interactive Streamlit front end for the five classifiers trained in
`train_models.py`. The app loads the pre-trained pipelines from `model/`
and evaluates whichever test CSV the user uploads. It NEVER retrains:
every number shown comes from applying a saved pipeline to the uploaded
data, exactly as required by the assignment.

Features
--------
1. Upload a test CSV (must contain the 30 feature columns + target column).
2. Pick a model from a dropdown (or compare all five at once).
3. See the six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC).
4. Inspect a confusion matrix and the full classification report.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

TARGET_DISPLAY = {0: "malignant", 1: "benign"}

st.set_page_config(
    page_title="Breast Cancer Classifier Explorer",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load all saved pipelines once and keep them in memory."""
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_resource(show_spinner=False)
def load_metadata() -> dict:
    """Load feature order / target info saved during training."""
    return joblib.load(MODEL_DIR / "feature_metadata.joblib")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def positive_scores(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Positive-class scores for AUC (probabilities or decision function)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def compute_metrics(pipeline, X: pd.DataFrame, y_true: pd.Series) -> dict:
    """Six evaluation metrics for a single fitted pipeline on uploaded data."""
    y_pred = pipeline.predict(X)
    scores = positive_scores(pipeline, X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def split_features_target(df: pd.DataFrame, feature_names: list[str], target_col: str):
    """Validate the upload and return (X, y) aligned to training feature order."""
    missing_target = target_col not in df.columns
    missing_features = [c for c in feature_names if c not in df.columns]
    return missing_target, missing_features


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    labels = [TARGET_DISPLAY.get(0, "0"), TARGET_DISPLAY.get(1, "1")]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("🔬 Breast Cancer Wisconsin — Multi-Model Classifier Explorer")
    st.caption(
        "Upload a test CSV, pick a classifier, and inspect its metrics. "
        "Models are pre-trained and loaded from disk — nothing is retrained here."
    )

    # Guard: models must exist
    if not MODEL_DIR.exists() or not all(
        (MODEL_DIR / f).exists() for f in MODEL_FILES.values()
    ):
        st.error(
            "Saved models were not found in `model/`. "
            "Run `python train_models.py` first to generate them."
        )
        st.stop()

    models = load_models()
    metadata = load_metadata()
    feature_names = metadata["feature_names"]
    target_col = metadata["target_column"]

    # --- Sidebar controls ---------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Controls")
        st.markdown(
            f"**Expected target column:** `{target_col}`  \n"
            f"**Expected features:** {len(feature_names)} numeric columns"
        )
        choice = st.selectbox(
            "Select model",
            ["— Compare all models —"] + list(MODEL_FILES.keys()),
            index=0,
        )
        st.divider()
        st.markdown(
            "**Tip:** use the `test_data.csv` from the repository. It already "
            "has the 30 feature columns and the `diagnosis` target column."
        )

    # --- Upload -------------------------------------------------------------
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    if uploaded is None:
        st.info("Awaiting a CSV upload to begin evaluation.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV: {exc}")
        return

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"Rows: **{len(df)}**  |  Columns: **{df.shape[1]}**")

    # --- Validate -----------------------------------------------------------
    missing_target, missing_features = split_features_target(df, feature_names, target_col)
    if missing_target:
        st.error(
            f"The uploaded CSV is missing the target column `{target_col}`. "
            "The app needs ground-truth labels to compute evaluation metrics."
        )
        return
    if missing_features:
        st.error(
            "The uploaded CSV is missing required feature columns: "
            f"{', '.join(missing_features[:8])}"
            + (" ..." if len(missing_features) > 8 else "")
        )
        return

    X = df[feature_names].copy()
    y_true = df[target_col].copy()

    # --- Evaluate -----------------------------------------------------------
    if choice == "— Compare all models —":
        st.subheader("📊 Model comparison on uploaded data")
        rows = {}
        for name, pipeline in models.items():
            rows[name] = compute_metrics(pipeline, X, y_true)
        comparison = pd.DataFrame(rows).T.round(4)
        st.dataframe(
            comparison.style.highlight_max(axis=0, color="#c6efce"),
            use_container_width=True,
        )
        winner = comparison["MCC"].idxmax()
        st.success(f"🏆 Best model on this data by MCC: **{winner}**")

        st.subheader("Confusion matrices")
        cols = st.columns(len(models))
        for col, (name, pipeline) in zip(cols, models.items()):
            with col:
                y_pred = pipeline.predict(X)
                st.pyplot(render_confusion_matrix(y_true, y_pred, name))
    else:
        pipeline = models[choice]
        metrics = compute_metrics(pipeline, X, y_true)
        y_pred = pipeline.predict(X)

        st.subheader(f"📈 Evaluation metrics — {choice}")
        m = st.columns(6)
        for col, (label, value) in zip(m, metrics.items()):
            col.metric(label, f"{value:.4f}")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(render_confusion_matrix(y_true, y_pred, choice))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y_true, y_pred,
                target_names=[TARGET_DISPLAY[0], TARGET_DISPLAY[1]],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).T.round(4),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
