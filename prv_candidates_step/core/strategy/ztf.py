import pickle

from .utils.ztf_prv_detections_parser import ZTFPreviousDetectionsParser
from .utils.ztf_non_detections_parser import ZTFNonDetectionsParser


def extract_detections_and_non_detections(alert):
    detections = [alert]
    non_detections = []

    prv_candidates = alert["extra_fields"]["prv_candidates"]
    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []

    aid, oid, parent = alert["aid"], alert["oid"], alert["candid"]
    for candidate in prv_candidates:
        if candidate["candid"]:
            candidate = ZTFPreviousDetectionsParser.parse_to_dict(candidate, aid, oid, parent)
            detections.append(candidate)
        else:
            candidate = ZTFNonDetectionsParser.parse_non_detection(candidate, aid, oid)
            non_detections.append(candidate)

    alert["extra_fields"].pop("prv_candidates")
    alert["extra_fields"]["parent_candid"] = None

    return {"aid": alert["aid"], "detections": detections, "non_detections": non_detections}
