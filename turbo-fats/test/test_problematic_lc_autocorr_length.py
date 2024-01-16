import unittest
import pandas as pd

from ztf_pipeline.features import FeaturesComputer

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.detections = pd.read_pickle('problematic_curve.pkl')
        self.features_computer = FeaturesComputer(0)

    def testProblematicCurves(self):
        result = self.features_computer.execute(self.detections)

if __name__ == '__main__':
    unittest.main()
