
import os
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report
)



MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target          # 0 = malignant, 1 = benign
    print(f"Dataset: Breast Cancer Wisconsin")
    print(f"  Rows: {len(X)}, Features: {X.shape[1]}, Target: binary (0=malignant, 1=benign)")
    print(f"  Class counts:\n{y.value_counts().to_string()}")
    return X, y



def evaluate(name, model, X_test_sc, y_test):
    y_pred = model.predict(X_test_sc)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_sc)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_sc)
    else:
        y_score = y_pred

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_score), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
    return metrics


def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))
    joblib.dump("target", os.path.join(MODEL_DIR, "target_col.joblib"))
    print(f"\nScaler and metadata saved.")

    # Save test_data.csv — features + target column, used by the Streamlit app
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"test_data.csv saved: {len(test_df)} rows, {len(test_df.columns)} columns")

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            "logistic_regression.joblib",
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            "decision_tree.joblib",
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            "knn.joblib",
        ),
        "Naive Bayes": (
            GaussianNB(),
            "naive_bayes.joblib",
        ),
        "Support Vector Machine": (
            CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
            "svm.joblib",
        ),
    }

    results = []
    for name, (model, filename) in models.items():
        model.fit(X_train_sc, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        print(f"\n{name} saved → model/{filename}")
        m = evaluate(name, model, X_test_sc, y_test)
        results.append(m)

    print("\n\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    df_results = pd.DataFrame(results).set_index("Model")
    print(df_results.to_string())
    df_results.to_csv("metrics_summary.csv")


if __name__ == "__main__":
    main()




