import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Validate outputs
assert len(pred) == len(y_test)
assert len(prob) == len(y_test)
assert 0 <= accuracy_score(y_test, pred) <= 1
assert confusion_matrix(y_test, pred).shape == (2, 2)
assert 0 <= roc_auc_score(y_test, prob) <= 1
assert not np.isnan(prob).any()

print("All W3D2 Logistic Regression tests passed successfully!")