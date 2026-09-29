import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ------------------------------------------
# LOAD DATA
# ------------------------------------------

housing = fetch_california_housing(
    as_frame=True
)

df = housing.frame

X = df.drop(
    columns=["MedHouseVal"]
)

y = df["MedHouseVal"]


# ------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------
# MODEL
# ------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# ------------------------------------------
# TRAIN
# ------------------------------------------

model.fit(
    X_train,
    y_train
)


# ------------------------------------------
# PREDICT
# ------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------
# TESTS
# ------------------------------------------

assert len(y_pred) == len(y_test)

assert mean_squared_error(
    y_test,
    y_pred
) >= 0

assert mean_absolute_error(
    y_test,
    y_pred
) >= 0

assert r2_score(
    y_test,
    y_pred
) <= 1

assert not np.isnan(y_pred).any()


print(
    "All W3D1 Linear Regression tests passed successfully!"
)