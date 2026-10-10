
"""W3D5: Compare Grid Search and Random Search for SVM tuning."""

from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def main():
    # Load data and reserve a final test set.
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Scaling is fitted inside each CV training fold.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ])

    parameter_grid = {
        "model__kernel": ["linear", "rbf"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", "auto"],
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    # Grid Search evaluates all 12 combinations.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    start = perf_counter()
    grid_search.fit(X_train, y_train)
    grid_seconds = perf_counter() - start

    # Random Search samples 8 combinations from the same space.
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=8,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )

    start = perf_counter()
    random_search.fit(X_train, y_train)
    random_seconds = perf_counter() - start

    # Evaluate the selected models on the same untouched test set.
    searches = {
        "Grid Search": (grid_search, grid_seconds),
        "Random Search": (random_search, random_seconds),
    }

    results = []

    for name, (search, elapsed) in searches.items():
        predictions = search.best_estimator_.predict(X_test)

        results.append({
            "search_method": name,
            "best_cv_accuracy": search.best_score_,
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_f1_macro": f1_score(
                y_test, predictions, average="macro"
            ),
            "fit_time_seconds": elapsed,
            "parameter_combinations_tested": len(search.cv_results_["params"]),
            "best_parameters": str(search.best_params_),
        })

        print(f"\n--- {name} ---")
        print("Best parameters:", search.best_params_)
        print("Best CV accuracy:", round(search.best_score_, 4))
        print("Test accuracy:", round(
            accuracy_score(y_test, predictions), 4
        ))
        print("Test macro F1:", round(
            f1_score(y_test, predictions, average="macro"), 4
        ))
        print("Search time (seconds):", round(elapsed, 2))
        print(classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
            zero_division=0,
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "W3D5_search_comparison.csv", index=False
    )

    # Save every candidate's cross-validation results for analysis.
    pd.DataFrame(grid_search.cv_results_).to_csv(
        OUTPUT_DIR / "W3D5_grid_search_details.csv", index=False
    )
    pd.DataFrame(random_search.cv_results_).to_csv(
        OUTPUT_DIR / "W3D5_random_search_details.csv", index=False
    )

    # Visualize test accuracy and search time.
    ax = results_df.set_index("search_method")[
        ["best_cv_accuracy", "test_accuracy"]
    ].plot(kind="bar", ylim=(0, 1.05), figsize=(8, 5))

    ax.set_title("Grid Search vs Random Search — Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Search method")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "W3D5_accuracy_comparison.png", dpi=160
    )
    plt.close()

    ax = results_df.plot(
        x="search_method",
        y="fit_time_seconds",
        kind="bar",
        legend=False,
        figsize=(8, 5),
    )
    ax.set_title("Hyperparameter Search Runtime")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("Search method")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "W3D5_runtime_comparison.png", dpi=160
    )
    plt.close()

    print("\n--- Final comparison ---")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved results to:", OUTPUT_DIR)
    print("W3D5 completed successfully.")


if __name__ == "__main__":
    main()
