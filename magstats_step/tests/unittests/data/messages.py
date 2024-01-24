import random
import pathlib

from fastavro import schema
from fastavro import utils

SCHEMA_PATH = str(
    pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent.parent,
        "schemas/correction_step",
        "output.avsc",
    )
)

SCHEMA = schema.load_schema(SCHEMA_PATH)
random.seed(42)

aids_pool = [f"AID22X{i}" for i in range(10)]
oids_pool = [f"ZTF{i}llmn" for i in range(50)]

data = list(utils.generate_many(SCHEMA, 10))
for d in data:
    aid = random.choice(aids_pool)
    oid = random.choice(oids_pool)
    d["oid"] = oid
    sid = "ZTF" if random.random() < 0.5 else "ATLAS"
    for detection in d["detections"]:
        detection["aid"] = aid
        detection["oid"] = oid
        detection["sid"] = sid
        detection["fid"] = "g" if random.random() < 0.5 else "r"
        detection["forced"] = False

    for non_detection in d["non_detections"]:
        non_detection["aid"] = aid
        non_detection["oid"] = oid
        non_detection["sid"] = sid
        non_detection["fid"] = "g" if random.random() < 0.5 else "r"
