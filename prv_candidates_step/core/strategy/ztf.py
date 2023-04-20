import copy
import pickle
from dataclasses import asdict

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.core.generic import GenericNonDetection
from survey_parser_plugins.core.mapper import Mapper
from survey_parser_plugins.parsers.ZTFParser import ZTFParser


def prv_detections_mapper() -> dict:
    mapping = copy.deepcopy(ZTFParser._mapping)
    mapping["oid"] = Mapper(lambda: ZTFPreviousDetectionsParser._oid)
    return mapping


def prv_non_detections_mapper() -> dict:
    mapping = copy.deepcopy(ZTFParser._mapping)
    mapping["oid"] = Mapper(lambda: ZTFNonDetectionsParser._oid)
    mapping["diffmaglim"] = Mapper(origin="diffmaglim")
    remove = ["candid", "pid", "ra", "e_ra", "dec", "e_dec", "mag", "e_mag", "isdiffpos"]
    for field in remove:
        mapping.pop(field)
    return mapping


class ZTFPreviousDetectionsParser(SurveyParser):
    _oid = ""
    _mapping = prv_detections_mapper()

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFPreviousDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, aid: str, oid: str, parent: str | int) -> dict:
        cls._oid = oid
        alert = asdict(cls.parse_message(message))
        alert["aid"] = aid
        alert["extra_fields"]["parent_candid"] = parent
        alert["has_stamp"] = False
        alert.pop("stamps")
        return alert


class ZTFNonDetectionsParser(SurveyParser):
    _oid = ""
    _mapping = prv_non_detections_mapper()
    _Model = GenericNonDetection

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFNonDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, aid: str, oid: str) -> dict:
        cls._oid = oid
        alert = asdict(cls.parse_message(message))
        alert["aid"] = aid
        alert.pop("stamps")
        alert.pop("extra_fields")
        return alert


def extract_detections_and_non_detections(alert: dict) -> dict:
    detections = [alert]
    non_detections = []

    prv_candidates = alert["extra_fields"]["prv_candidates"]
    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []

    aid, oid, parent = alert["aid"], alert["oid"], alert["candid"]
    for candidate in prv_candidates:
        if candidate["candid"]:
            candidate = ZTFPreviousDetectionsParser.parse(candidate, aid, oid, parent)
            detections.append(candidate)
        else:
            candidate = ZTFNonDetectionsParser.parse(candidate, aid, oid)
            non_detections.append(candidate)

    alert["extra_fields"].pop("prv_candidates")
    alert["extra_fields"]["parent_candid"] = None

    return {"aid": alert["aid"], "detections": detections, "non_detections": non_detections}
