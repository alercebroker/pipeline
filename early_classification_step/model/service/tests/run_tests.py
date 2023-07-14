import pandas as pd
import requests
import json
import io
import unittest
import time


IP = "0.0.0.0"
# IP = '18.191.43.15'


class TestCLFFeaturesMethods(unittest.TestCase):
    def testClf(self):
        session = requests.Session()
        input_data = pd.read_pickle("../../tests/small_test_with_features.pkl")
        for i, test in input_data.iterrows():
            t = time.time()
            metadata_columns = [
                c
                for c in test.keys()
                if c not in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
            ]
            files = {
                "cutoutScience": io.BytesIO(test["cutoutScience"]),
                "cutoutTemplate": io.BytesIO(test["cutoutTemplate"]),
                "cutoutDifference": io.BytesIO(test["cutoutDifference"]),
            }

            metadata = io.StringIO()
            metadata_df = pd.DataFrame(test[metadata_columns]).transpose().copy()
            metadata_df.set_index("oid", inplace=True)
            metadata_df.to_csv(metadata)
            files["metadata"] = metadata.getvalue()

            resp = session.post(f"http://{IP}:5000/get_classification", files=files)
            if resp.status_code == 200:
                probs = resp.content
                probs = json.loads(probs)
                for key in probs:
                    self.assertEqual(type(probs[key]), float)
                    self.assertTrue(probs[key] >= 0.0)
                    self.assertTrue(probs[key] <= 1.0)
                print(f"{time.time() - t:.3f} [s]", probs)

            elif resp.status_code == 500:
                status = resp.content
                status = json.loads(status)
                self.assertEqual(status["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
