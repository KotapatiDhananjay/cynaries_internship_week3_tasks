
"""W3D3: Decision Trees and Random Forests.

Train and compare classifiers, visualize a decision tree, and tune
hyperparameters to reduce overfitting.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def main():
    # 1. Load a built-in binary classification dataset.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Dataset shape:", X.shape)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Classes:", list(dataset.target_names))

    # 2. Train an unrestricted tree as a baseline.
    baseline_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_tree.fit(X_train, y_train)

    baseline_train_acc = accuracy_score(
        y_train, baseline_tree.predict(X_train)
    )
    baseline_test_acc = accuracy_score(
        y_test, baseline_tree.predict(X_test)
    )

    print("\n--- Baseline Decision Tree ---")
    print("Training accuracy:", round(baseline_train_acc, 4))
    print("Testing accuracy:", round(baseline_test_acc, 4))
    print("Tree depth:", baseline_tree.get_depth())
    print("Number of leaves:", baseline_tree.get_n_leaves())

    # 3. Tune tree complexity with stratified 5-fold cross-validation.
    # Gini measures class impurity; entropy uses information theory.
    parameter_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_tree = search.best_estimator_

    # 4. Train a Random Forest for comparison.
    forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)

    # 5. Compare models on training and held-out testing data.
    models = {
        "Baseline Decision Tree": baseline_tree,
        "Tuned Decision Tree": tuned_tree,
        "Random Forest": forest,
    }

    rows = []
    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        rows.append({
            "model": name,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_gap": train_acc - test_acc,
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "W3D3_model_comparison.csv", index=False)

    print("\n--- Model Comparison ---")
    print(results.round(4).to_string(index=False))
    print("\nBest Decision Tree parameters:", search.best_params_)
    print("Best CV accuracy:", round(search.best_score_, 4))

    # 6. Visualize the tuned tree.
    plt.figure(figsize=(22, 12))
    plot_tree(
        tuned_tree,
        feature_names=list(X.columns),
        class_names=list(dataset.target_names),
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Tuned Decision Tree (display limited to first 3 levels)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "W3D3_decision_tree.png", dpi=180)
    plt.close()

    # 7. Plot training and testing accuracy.
    ax = results.set_index("model")[
        ["train_accuracy", "test_accuracy"]
    ].plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))

    ax.set_title("Training vs Testing Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "W3D3_accuracy_comparison.png", dpi=180)
    plt.close()

    # 8. Plot confusion matrix for the tuned tree.
    predictions = tuned_tree.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=dataset.target_names,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Tuned Decision Tree — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "W3D3_confusion_matrix.png", dpi=180)
    plt.close()

    # 9. Export feature importance for interpretation.
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": tuned_tree.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / "W3D3_feature_importance.csv", index=False
    )

    print("\n--- Tuned Decision Tree Classification Report ---")
    print(classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
        zero_division=0,
    ))

    print("Top 10 important features:")
    print(importance.head(10).to_string(index=False))

    print("\nOutput files saved in:", OUTPUT_DIR)
    print("W3D3 project completed successfully.")


if __name__ == "__main__":
    main()
