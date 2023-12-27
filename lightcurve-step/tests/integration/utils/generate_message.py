from fastavro.utils import generate_many
import random

random.seed(42)


def get_parent_candid(index, candids):
    if random.random() < 0.5:
        try:
            return candids[index]
        except IndexError:
            return None


def get_candid(index, candids):
    try:
        return candids[index]
    except IndexError:
        return random.randint(1000000, 10000000)


def generate_message(schema, n=10, detections=[], aids=[]):
    messages = generate_many(schema, n)
    parsed_msgs = []
    for i, message in enumerate(messages):
        aid = aids[i]
        message["aid"] = aid
        message["detections"] = detections
        parsed_msgs.append(message)

    return parsed_msgs
