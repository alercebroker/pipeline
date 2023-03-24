import random

from fastavro import schema
from fastavro import utils

loaded = schema.load_schema("schema.avsc")


def extra_fields_generator():
    extra_fields_schema = {
        'doc': 'Extra Fields',
        'name': 'ExtraFields',
        'type': 'record',
        'fields': [
            {'name': 'distnr', 'type': 'float'},
            {'name': 'distpsnr1', 'type': 'float'},
            {'name': 'sgscore1', 'type': 'float'},
            {'name': 'chinr', 'type': 'float'},
            {'name': 'sharpnr', 'type': 'float'},
            # TODO: For the magap statistics. Not sure if this is the actual name.
            {'name': 'mjd', 'type': 'float'},
            {'name': 'mag', 'type': 'float'},
            {'name': 'e_mag', 'type': 'float'},
        ],
    }
    return utils.generate_one(extra_fields_schema)


data = list(utils.generate_many(loaded, 10))
for d in data:
    for detection in d['detections']:
        detection["fid"] = 1 if random.random() < .5 else 2
        detection["extra_fields"] = extra_fields_generator()
    for non_detection in d["non_detections"]:
        non_detection["fid"] = 1 if random.random() < .5 else 2
