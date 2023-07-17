import pandas as pd
import io
import requests
import json


IP = "0.0.0.0"
input_data = pd.read_pickle("../../tests/small_test_with_features.pkl")
counter = 0
for i, test in input_data.iterrows():
    for _ in range(5):
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

        resp = requests.post(f"http://{IP}:5000/get_classification", files=files)
        counter += 1
        if resp.status_code == 200:
            probs = resp.content
            probs = json.loads(probs)
            print(probs)
        print(resp.status_code)
        print(counter)
requests.post(f"http://{IP}:5000/get_classification", json="save")
