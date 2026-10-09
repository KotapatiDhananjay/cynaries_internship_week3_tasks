
"""W3D4: Compare and tune SVM and KNN classifiers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def main():
    # Load dataset and split before fitting any preprocessing.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Scaling is essential for distance- and margin-based models.
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ])

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ])

    # Tune parameters using training data only.
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    svm_search = GridSearchCV(
        svm_pipeline,
        {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"],
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    knn_search = GridSearchCV(
        knn_pipeline,
        {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],  # Manhattan or Euclidean distance
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    searches = {
        "SVM": svm_search,
        "KNN": knn_search,
    }

    rows = []
    fitted_models = {}

    for name, search in searches.items():
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fitted_models[name] = model

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        rows.append({
            "model": name,
            "train_accuracy": accuracy_score(
                y_train, train_predictions
            ),
            "test_accuracy": accuracy_score(
                y_test, test_predictions
            ),
            "test_f1_macro": f1_score(
                y_test, test_predictions, average="macro"
            ),
            "best_cv_accuracy": search.best_score_,
            "best_parameters": str(search.best_params_),
        })

        print(f"\n--- {name} ---")
        print("Best parameters:", search.best_params_)
        print("Best CV accuracy:", round(search.best_score_, 4))
        print("Test accuracy:", round(
            accuracy_score(y_test, test_predictions), 4
        ))
        print(classification_report(
            y_test,
            test_predictions,
            target_names=dataset.target_names,
            zero_division=0,
        ))

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_DIR / "W3D4_model_comparison.csv", index=False
    )
    print("\n--- Model Comparison ---")
    print(results.round(4).to_string(index=False))

    # Plot test accuracy for both tuned models.
    ax = results.set_index("model")["test_accuracy"].plot(
        kind="bar", figsize=(8, 5), ylim=(0, 1.05)
    )
    ax.set_title("Tuned SVM vs KNN — Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "W3D4_accuracy_comparison.png", dpi=160
    )
    plt.close()

    # Confusion matrix for the model with the best test accuracy.
    # This selection is for displaying evidence only; don't use the
    # test set to tune hyperparameters.
    best_name = results.loc[
        results["test_accuracy"].idxmax(), "model"
    ]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        display_labels=dataset.target_names,
        cmap="Blues",
    )
    plt.title(f"{best_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "W3D4_confusion_matrix.png", dpi=160
    )
    plt.close()

    # Permutation importance works with both SVM and KNN.
    # It measures the drop in score when a feature is shuffled.
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "W3D4_permutation_importance.csv",
        index=False,
    )

    top_features = importance_df.head(10).sort_values(
        "importance_mean"
    )
    ax = top_features.plot(
        x="feature",
        y="importance_mean",
        kind="barh",
        legend=False,
        figsize=(9, 6),
    )
    ax.set_title(f"Top 10 Permutation Importances — {best_name}")
    ax.set_xlabel("Mean decrease in accuracy after shuffling")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "W3D4_feature_importance.png", dpi=160
    )
    plt.close()

    print("\nFeature importance saved.")
    print("Selected for display:", best_name)
    print("\nOutput folder:", OUTPUT_DIR)
    print("W3D4 project completed successfully.")


if __name__ == "__main__":
    main()
