import random

from fastavro import schema
from fastavro import utils

loaded = schema.load_schema("schema.avsc")


data = list(utils.generate_many(loaded, 10))
for d in data:
    for detection in d['detections']:
        detection["aid"] = "AID1" if random.random() < .5 else "AID2"
        detection["fid"] = 1 if random.random() < .5 else 2
    for non_detection in d["non_detections"]:
        non_detection["aid"] = "AID1" if random.random() < .5 else "AID2"
        non_detection["fid"] = 1 if random.random() < .5 else 2
