import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ==========================================
# W3D1 - LINEAR REGRESSION
# ==========================================

print("=" * 60)
print("W3D1 - LINEAR REGRESSION")
print("=" * 60)


# ------------------------------------------
# 1. LOAD REAL DATASET
# ------------------------------------------

housing = fetch_california_housing(
    as_frame=True
)

df = housing.frame

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------
# 2. CHECK DATA
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Statistics:")
print(df.describe())


# ------------------------------------------
# 3. FEATURES AND TARGET
# ------------------------------------------

X = df.drop(
    columns=["MedHouseVal"]
)

y = df["MedHouseVal"]


# ------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ------------------------------------------
# 5. DEFINE MODELS
# ------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001))
    ])
}


# ------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# ------------------------------------------

results = []

predictions = {}

for name, model in models.items():

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test
    )

    predictions[name] = y_pred

    # Metrics
    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


# ------------------------------------------
# 7. RESULTS TABLE
# ------------------------------------------

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------
# 8. PRINT LINEAR REGRESSION COEFFICIENTS
# ------------------------------------------

linear_model = models[
    "Linear Regression"
]

coefficients = linear_model.named_steps[
    "model"
].coef_

coefficient_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
})

print("\nLinear Regression Coefficients:")
print(coefficient_df)


# ------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------

results_df.to_csv(
    "W3D1_model_comparison.csv",
    index=False
)

coefficient_df.to_csv(
    "W3D1_linear_coefficients.csv",
    index=False
)


# ------------------------------------------
# 10. PREDICTED VS ACTUAL
# ------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions["Linear Regression"],
    alpha=0.5
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Predicted vs Actual")

plt.tight_layout()

plt.savefig(
    "W3D1_predicted_vs_actual.png"
)

plt.show()


# ------------------------------------------
# 11. RESIDUAL PLOT
# ------------------------------------------

residuals = (
    y_test -
    predictions["Linear Regression"]
)

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions["Linear Regression"],
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression - Residual Plot")

plt.tight_layout()

plt.savefig(
    "W3D1_residuals.png"
)

plt.show()


# ------------------------------------------
# 12. SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)
print("W3D1 SUMMARY")
print("=" * 60)

print("""
1. Loaded the California Housing dataset.
2. Split the data into training and testing sets.
3. Standardized features using StandardScaler.
4. Trained Linear Regression, Ridge, and Lasso.
5. Evaluated models using MSE, RMSE, MAE, and R².
6. Compared the three models.
7. Printed Linear Regression coefficients.
8. Created predicted-vs-actual and residual plots.
9. Saved model comparison and coefficient results.
""")

print("\nW3D1 completed successfully!")