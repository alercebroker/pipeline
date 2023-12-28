from typing import List


def extract_detections_from_messages(messages: List[dict]):
    aids = {}
    for msg in messages:
        dets = msg["detections"]
        non_dets = msg["non_detections"]
        oids = list(
            set([dt["oid"] for dt in dets + non_dets if dt["sid"] == "ZTF"])
        )
        aids[msg["aid"]] = {
            "oid": oids,
            "detections": dets,
            "non_detections": non_dets,
        }

    return aids
