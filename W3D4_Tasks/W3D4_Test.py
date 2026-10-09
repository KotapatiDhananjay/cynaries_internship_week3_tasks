
"""Validation tests for W3D4 SVM and KNN."""

import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TestSVMKNN(unittest.TestCase):

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

    def test_svm_predictions(self):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ])
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.70
        )

    def test_knn_predictions(self):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ])
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.y_test))
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.70
        )

    def test_scaling_is_in_pipeline(self):
        for model in [
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC()),
            ]),
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
        ]:
            self.assertIsInstance(
                model.named_steps["scaler"], StandardScaler
            )

    def test_reproducible_split(self):
        X1, X2, y1, y2 = train_test_split(
            self.X_train,
            self.y_train,
            test_size=0.25,
            random_state=42,
            stratify=self.y_train,
        )
        X3, X4, y3, y4 = train_test_split(
            self.X_train,
            self.y_train,
            test_size=0.25,
            random_state=42,
            stratify=self.y_train,
        )

        self.assertTrue(X1.equals(X3))
        self.assertTrue(X2.equals(X4))
        self.assertTrue(y1.equals(y3))
        self.assertTrue(y2.equals(y4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
