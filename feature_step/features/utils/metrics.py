from typing import List


def get_sid(list_of_objects: List[list]):
    sids = []
    for obj in list_of_objects:
        obj_sids = []
        for det in obj["detections"]:
            obj_sids.append(det["sid"])
        sids.extend(obj_sids)
    return list(set(sids))
