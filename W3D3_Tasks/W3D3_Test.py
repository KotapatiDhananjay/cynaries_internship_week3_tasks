
"""Basic validation tests for W3D3 Decision Trees and Random Forests."""

import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class TestW3D3Models(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer(as_frame=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = (
            train_test_split(
                data.data,
                data.target,
                test_size=0.20,
                random_state=42,
                stratify=data.target,
            )
        )

    def test_decision_tree_predictions(self):
        model = DecisionTreeClassifier(
            max_depth=4, random_state=42
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.70
        )

    def test_random_forest_predictions(self):
        model = RandomForestClassifier(
            n_estimators=50, random_state=42
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.y_test))
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.70
        )

    def test_tree_depth_limit(self):
        model = DecisionTreeClassifier(
            max_depth=3, random_state=42
        )
        model.fit(self.X_train, self.y_train)

        self.assertLessEqual(model.get_depth(), 3)

    def test_feature_importances(self):
        model = DecisionTreeClassifier(
            max_depth=4, random_state=42
        )
        model.fit(self.X_train, self.y_train)

        self.assertEqual(
            len(model.feature_importances_), self.X_train.shape[1]
        )
        self.assertAlmostEqual(
            model.feature_importances_.sum(), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
