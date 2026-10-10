
"""Basic tests for W3D5 hyperparameter tuning."""

import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TestHyperparameterTuning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer(as_frame=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = (
            train_test_split(
                data.data,
                data.target,
                test_size=0.20,
                stratify=data.target,
                random_state=42,
            )
        )

        cls.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ])

        cls.parameters = {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1],
            "model__gamma": ["scale", "auto"],
        }

        cls.cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=42
        )

    def test_grid_search(self):
        search = GridSearchCV(
            self.pipeline,
            self.parameters,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)

        predictions = search.best_estimator_.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.70
        )

    def test_random_search(self):
        search = RandomizedSearchCV(
            self.pipeline,
            self.parameters,
            n_iter=4,
            cv=self.cv,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)

        predictions = search.best_estimator_.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.70
        )

    def test_scaling_is_inside_pipeline(self):
        self.assertIsInstance(
            self.pipeline.named_steps["scaler"], StandardScaler
        )

    def test_search_finds_best_parameters(self):
        search = GridSearchCV(
            self.pipeline,
            self.parameters,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)

        self.assertIn("model__C", search.best_params_)
        self.assertIn("model__kernel", search.best_params_)
        self.assertGreaterEqual(search.best_score_, 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
