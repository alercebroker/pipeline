import copy
import pickle
from dataclasses import asdict

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.core.generic import GenericNonDetection
from survey_parser_plugins.core.mapper import Mapper
from survey_parser_plugins.parsers import ZTFParser


def prv_non_detections_mapper() -> dict:
    mapping = copy.deepcopy(ZTFParser._mapping)
    preserve = ["oid", "sid", "tid", "fid", "mjd"]

    mapping = {k: v for k, v in mapping.items() if k in preserve}
    mapping.update({"diffmaglim": Mapper(origin="diffmaglim")})
    return mapping


class ZTFPreviousDetectionsParser(SurveyParser):
    _mapping = ZTFParser._mapping

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFPreviousDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, oid: str) -> dict:
        message = message.copy()
        message["objectId"] = oid
        return asdict(cls.parse_message(message))


class ZTFNonDetectionsParser(SurveyParser):
    _mapping = prv_non_detections_mapper()
    _Model = GenericNonDetection

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFNonDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, oid: str) -> dict:
        message = message.copy()
        message["objectId"] = oid
        return asdict(cls.parse_message(message))


def extract_detections_and_non_detections(alert: dict) -> dict:
    detections = [alert]
    non_detections = []

    prv_candidates = alert["extra_fields"]["prv_candidates"]
    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []

    aid, oid, parent = alert["aid"], alert["oid"], alert["candid"]
    for candidate in prv_candidates:
        if candidate["candid"]:
            candidate = ZTFPreviousDetectionsParser.parse(candidate, oid)
            candidate.update({"aid": aid, "has_stamp": False})
            candidate["extra_fields"].update({"parent_candid": parent})
            candidate.pop("stamps", None)
            detections.append(candidate)
        else:
            candidate = ZTFNonDetectionsParser.parse(candidate, oid)
            candidate.update({"aid": aid})
            candidate.pop("stamps", None)
            candidate.pop("extra_fields", None)
            non_detections.append(candidate)

    alert["extra_fields"].pop("prv_candidates")
    alert["extra_fields"]["parent_candid"] = None

    return {"aid": alert["aid"], "detections": detections, "non_detections": non_detections, "forced_photometries": []}
