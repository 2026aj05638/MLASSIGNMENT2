# ML Assignment 2 - Breast Cancer Classification

## a. Problem Statement
This project trains five supervised machine learning classification models to predict whether a tumour is malignant or benign based on 30 numeric features computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. The goal is to compare model performance across six standard evaluation metrics and deploy the models in an interactive web application using Streamlit.


## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository

| Property | Value |
|---|---|
| Source | UCI ML Repository (built into scikit-learn) |
| Rows | 569 |
| Features | 30 numeric features |
| Target | Binary — `0` = Malignant, `1` = Benign |
| Task | Binary Classification |
| Missing values | None |

Features are computed from a digitised image of a fine needle aspirate of a breast mass. They describe characteristics of the cell nuclei present in the image — including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each measured as mean, standard error, and worst (largest) value across the image, yielding 30 features total.


## c. GitHub Repository Link


Repository contents:
```
MLASSIGNMENT2/
  app.py                    ← Streamlit web application
  train_models.py           ← Model training script
  requirements.txt          ← Python dependencies
  README.md                 ← This file
  test_data.csv             ← 114-row test split (20% of dataset)
  model/
    logistic_regression.joblib
    decision_tree.joblib
    knn.joblib
    naive_bayes.joblib
    svm.joblib
    scaler.joblib
    feature_columns.joblib
    target_col.joblib
```

## d. Models Used — Evaluation Metrics

All models trained on 455 rows (80%) and evaluated on 114 rows (20%).
Train/test split: stratified, `random_state=42`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |

*Metrics: Precision, Recall, F1 are weighted averages across both classes.*


## e. Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Excellent performance — 98.25% accuracy and the highest AUC (0.9954). The 30 features have strong linear separability, making logistic regression very effective. Tied with SVM as the top performer. |
| Decision Tree | Lowest performer overall — 91.23% accuracy and lowest AUC (0.9157). The unpruned tree overfits to training noise. Still a reasonable baseline but prone to high variance. |
| KNN | Good performance — 95.61% accuracy and AUC of 0.9788. Benefits strongly from StandardScaler since KNN is distance-based. Slower at inference than the other models but reliable. |
| Naive Bayes | Moderate — 92.98% accuracy but strong AUC (0.9868), meaning it ranks probabilities well even though its hard predictions are less precise. The feature independence assumption is violated but it still generalises adequately. |
| Support Vector Machine | Excellent — 98.25% accuracy and AUC of 0.9950, tied with Logistic Regression on all metrics. The RBF kernel (default) effectively captures non-linear boundaries in the 30-dimensional feature space. |
| **Overall, Winner for the dataset?** | **Logistic Regression / SVM (tied)** — both achieve 98.25% accuracy, 0.9825 F1, and MCC of 0.9623. Logistic Regression is preferred as the final recommendation due to its higher AUC (0.9954 vs 0.9950) and faster inference. |


## Live Streamlit App

