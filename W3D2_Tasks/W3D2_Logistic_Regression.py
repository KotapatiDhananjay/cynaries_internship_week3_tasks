import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.multiclass import OneVsRestClassifier


# ==========================================
# W3D2 - LOGISTIC REGRESSION & CLASSIFICATION
# ==========================================

print("=" * 60)
print("W3D2 - LOGISTIC REGRESSION & CLASSIFICATION")
print("=" * 60)


# ------------------------------------------
# 1. LOAD BINARY CLASSIFICATION DATASET
# ------------------------------------------

data = load_breast_cancer(as_frame=True)
df = data.frame

X = df.drop(columns=["target"])
y = df["target"]

print("\nDataset Shape:", df.shape)
print("\nClass Distribution:")
print(y.value_counts())


# ------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------
# 3. TRAIN LOGISTIC REGRESSION
# ------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ------------------------------------------
# 4. EVALUATION METRICS
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nBINARY CLASSIFICATION RESULTS")
print("-----------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------
# 5. PRINT COEFFICIENTS AND INTERCEPT
# ------------------------------------------

classifier = model.named_steps["classifier"]

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": classifier.coef_[0]
})

print("\nModel Intercept:")
print(classifier.intercept_[0])

print("\nFirst 10 Coefficients:")
print(coefficients.head(10).to_string(index=False))

coefficients.to_csv(
    "W3D2_binary_coefficients.csv",
    index=False
)


# ------------------------------------------
# 6. CONFUSION MATRIX HEATMAP
# ------------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Breast Cancer - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig("W3D2_confusion_matrix.png")
plt.show()


# ------------------------------------------
# 7. ROC CURVE
# ------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig("W3D2_roc_curve.png")
plt.show()


# ------------------------------------------
# 8. SAVE BINARY METRICS
# ------------------------------------------

binary_results = pd.DataFrame([{
    "Model": "Binary Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": auc
}])

binary_results.to_csv(
    "W3D2_binary_metrics.csv",
    index=False
)


# ------------------------------------------
# 9. MULTICLASS CLASSIFICATION WITH IRIS
# ------------------------------------------

iris = load_iris(as_frame=True)

X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris,
    y_iris,
    test_size=0.20,
    random_state=42,
    stratify=y_iris
)

multiclass_models = {
    "One-vs-Rest": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(max_iter=2000)
        ))
    ]),
    "Multinomial (Softmax)": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            max_iter=2000
        ))
    ])
}

multiclass_results = []

for name, clf in multiclass_models.items():
    clf.fit(X_train_i, y_train_i)

    pred = clf.predict(X_test_i)

    score = accuracy_score(y_test_i, pred)
    f1_macro = f1_score(
        y_test_i,
        pred,
        average="macro"
    )

    multiclass_results.append({
        "Model": name,
        "Accuracy": score,
        "Macro_F1": f1_macro
    })

    print(f"\n{name}")
    print(f"Accuracy: {score:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

multiclass_df = pd.DataFrame(multiclass_results)

print("\nMULTICLASS MODEL COMPARISON")
print(multiclass_df.to_string(index=False))

multiclass_df.to_csv(
    "W3D2_multiclass_comparison.csv",
    index=False
)


# ------------------------------------------
# 10. SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)
print("W3D2 SUMMARY")
print("=" * 60)

print("""
1. Trained Logistic Regression on breast cancer data.
2. Printed model coefficients and intercept.
3. Evaluated Accuracy, Precision, Recall, F1 and ROC-AUC.
4. Created a confusion matrix heatmap and ROC curve.
5. Compared One-vs-Rest and multinomial logistic regression
   on the Iris dataset.
6. Saved evaluation results as CSV files.
""")

print("\nW3D2 completed successfully!")