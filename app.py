#StreamLit App

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    target_col = joblib.load(os.path.join(MODEL_DIR, "target_col.joblib"))
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return scaler, feature_cols, target_col, models


def compute_metrics(model, scaler, X, y):
    X_sc = scaler.transform(X)
    y_pred = model.predict(X_sc)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_sc)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_sc)
    else:
        y_score = y_pred

    try:
        auc = round(float(roc_auc_score(y, y_score)), 4)
    except Exception:
        auc = 0.0

    return {
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "AUC": auc,
        "Precision": round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y, y_pred), 4),
    }, y_pred


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    return fig

# App Layout
st.set_page_config(page_title="Machine Learning Assignment 2", layout="wide")
st.title("Breast Cancer Classification — ML Assignment 2")
st.markdown("Upload the test CSV, select a model, and view evaluation results.")

try:
    scaler, feature_cols, target_col, models = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

#Sidebar
with st.sidebar:
    st.header("Settings")
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    st.markdown("---")
    st.markdown("**Required CSV columns:**")
    st.code("\n".join(feature_cols + [target_col]))


#FileUpload
st.subheader("1. Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing feature columns and the target column",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write(f"Uploaded: **{len(df)} rows × {len(df.columns)} columns**")
st.dataframe(df.head())

# Validate columns
missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    st.error(f"Missing feature columns: {missing_features}")
    st.stop()
if target_col not in df.columns:
    st.error(f"Target column `{target_col}` not found in uploaded file.")
    st.stop()

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[target_col].astype(int)



#Model Evaluation
st.subheader(f"2. Results — {selected_model_name}")
model = models[selected_model_name]

metrics, y_pred = compute_metrics(model, scaler, X, y)

st.subheader("3. Evaluation Metrics")
metrics_df = pd.DataFrame([metrics])
st.table(metrics_df.style.format("{:.4f}"))

col1, col2 = st.columns(2)

with col1:
    st.subheader("4. Confusion Matrix")
    fig = plot_confusion_matrix(y, y_pred, selected_model_name)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("4. Classification Report")
    report = classification_report(y, y_pred, output_dict=False)
    st.text(report)



#Compare all models
if st.checkbox("Compare all 5 models on this dataset"):
    st.subheader("All Models Comparison")
    all_metrics = []
    for name, m in models.items():
        met, _ = compute_metrics(m, scaler, X, y)
        met["Model"] = name
        all_metrics.append(met)
    comparison_df = pd.DataFrame(all_metrics).set_index("Model")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen")
                               .highlight_min(axis=0, color="lightyellow")
                               .format("{:.4f}"))
