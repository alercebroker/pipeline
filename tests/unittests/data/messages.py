import random

from fastavro import schema
from fastavro import utils

SCHEMA = schema.load_schema("schema.avsc")
random.seed(42)

aids_pool = [f"AID22X{i}" for i in range(10)]

data = list(utils.generate_many(SCHEMA, 10))
for d in data:
    aid = random.choice(aids_pool)
    d["aid"] = aid
    for detection in d["detections"]:
        detection["aid"] = aid
        detection["fid"] = 1 if random.random() < 0.5 else 2
    for non_detection in d["non_detections"]:
        non_detection["aid"] = aid
        non_detection["fid"] = 1 if random.random() < 0.5 else 2
