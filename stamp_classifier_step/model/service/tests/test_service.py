import pandas as pd
import requests
import json
import base64
import io

input_data = pd.read_pickle("../../tests/small_dataset.pkl")
test = input_data.sample(1).iloc[0]
files = {
    "cutoutScience": io.BytesIO(test["cutoutScience"]),
    "cutoutTemplate": io.BytesIO(test["cutoutTemplate"]),
    "cutoutDifference": io.BytesIO(test["cutoutDifference"]),
}
resp = requests.post("http://localhost:5000/get_classification", files=files)
print(resp.content)
# print(json.loads(resp.content))
