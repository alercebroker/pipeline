from fastavro.utils import generate_many
import random

random.seed(42)


def generate_message(schema, n=10):
    messages = generate_many(schema, n)
    parsed_msgs = []
    n_ztf, n_atlas = 0, 0
    # lets say 70% ztf, 30% atlas
    for i, message in enumerate(messages):
        aid = f"AL{i:02}XYZ{n_ztf:02}{n_atlas:02}"
        message["aid"] = aid
        dice = random.random()
        if dice < 0.7:
            sid, tid = "ZTF", "ZTF"
            oid = f"ZTF{n_ztf:03}llmn"
            n_ztf += 1
        else:
            sid, tid = "ATLAS", "ATLAS-01a"
            oid = f"ATLAS{n_atlas:03}kxnmas"
            n_atlas += 1
        for detection in message["detections"]:
            detection["sid"] = sid
            detection["tid"] = tid
            detection["oid"] = oid
            detection["aid"] = aid
            detection["candid"] = random.randint(1000000, 10000000)
            detection["fid"] = "g"
            detection["parent_candid"] = random.randint(1000000, 10000000)

        parsed_msgs.append(message)

    return parsed_msgs
