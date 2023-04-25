from typing import List


def extract_detections_from_messages(messages: List[dict]):
    return {
        msg["aid"]: {
            "detections": msg["detections"],
            "non_detections": msg["non_detections"],
        }
        for msg in messages
    }
