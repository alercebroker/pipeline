import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
import pandas as pd
from deployment.stamp_clf import StampClassifier

if __name__ == "__main__":
    clf = StampClassifier()
    input_data = pd.read_pickle("../tests/single_stamp.pkl")
    pred = clf.execute(input_data)
    print(pred)
    print(pred.shape)
