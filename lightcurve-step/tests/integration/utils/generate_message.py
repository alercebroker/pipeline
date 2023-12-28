from fastavro.utils import generate_many
import random

random.seed(42)


def generate_message(schema, n=10, detections=[], oids=[]):
    messages = generate_many(schema, n)
    parsed_msgs = []
    for i, message in enumerate(messages):
        oid = oids[i]
        message["oid"] = oid
        message["detections"] = detections
        parsed_msgs.append(message)

    return parsed_msgs
