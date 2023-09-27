import random

from fastavro import schema
from fastavro import utils

SCHEMA = schema.load_schema("schema.avsc")
random.seed(42)

aids_pool = [f"AID22X{i}" for i in range(50)]
oids_pool = [f"ZTF{i}llmn" for i in range(100)]

data = list(utils.generate_many(SCHEMA, 10))
for d in data:
    aid = random.choice(aids_pool)
    d["aid"] = aid
    oid = random.choice(oids_pool)
    sid = "ZTF" if random.random() < 0.5 else "ATLAS"
    for detection in d["detections"]:
        detection["aid"] = aid
        detection["sid"] = sid
        detection["fid"] = "g" if random.random() < 0.5 else "r"
        detection["forced"] = False
        detection["oid"] = oid
        
    if sid != "ZTF":
        d["non_detections"] = []
        continue
    for non_detection in d["non_detections"]:
        non_detection["aid"] = aid
        non_detection["sid"] = sid
        non_detection["oid"] = oid
        non_detection["fid"] = "g" if random.random() < 0.5 else "r"
