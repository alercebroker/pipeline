import unittest

import pandas as pd

from alerce_classifiers.dummy_classifier import DummyClassifier


class DummyClassifierTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DummyClassifier("this_path_doesnt_exists")
        self.input_data = pd.DataFrame({
            "feature_1": [0, 0, 0],
            "feature_2": [0, 0, 0]
        })

    def test_preprocess(self):
        preprocessed = self.model.preprocess(self.input_data)
        self.assertIsInstance(preprocessed, pd.DataFrame)
        self.assertListEqual(preprocessed.columns.to_list(), self.input_data.columns.to_list())

    def test_predict_proba(self):
        pass
