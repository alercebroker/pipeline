from fastavro import schema
from fastavro import utils

loaded = schema.load_schema("tests/unittest/data/detection_schema.avsc")


def extra_fields_generator():
    return {}


data = utils.generate_many(loaded, 10)
for d in data:
    d["extra_fields"] = extra_fields_generator()
