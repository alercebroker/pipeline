import copy
import pickle
from dataclasses import asdict

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.parsers import LSSTParser


def forced_photometry_mapper() -> dict:
    mapping = copy.deepcopy(LSSTParser._mapping)
    remove = ["pid", "ra", "dec", "e_ra", "e_dec"]

    mapping = {k: v for k, v in mapping.items() if k not in remove}
    return mapping


class LSSTPreviousDetectionsParser(SurveyParser):
    _mapping = LSSTParser._mapping

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(LSSTPreviousDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict) -> dict:
        return asdict(cls.parse_message(message))


class LSSTForcedPhotometryParser(SurveyParser):
    _mapping = forced_photometry_mapper()

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(LSSTForcedPhotometryParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict) -> dict:
        message = message.copy()
        message["diaSourceId"] = message.pop("diaForcedSourceId")
        return asdict(cls.parse_message(message))


def extract_detections_and_non_detections(alert: dict) -> dict:
    detections = [alert]
    forced_photometries = []

    aid, parent = alert["aid"], alert["candid"]

    prv_candidates = alert["extra_fields"].pop("prvDiaSources")
    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []
    for candidate in prv_candidates:
        candidate = LSSTPreviousDetectionsParser.parse(candidate)
        candidate.update({"aid": aid, "has_stamp": False})
        candidate["extra_fields"].update({"parent_candid": parent})
        candidate.pop("stamps", None)
        detections.append(candidate)

    prv_forced = alert["extra_fields"].pop("prvDiaForcedSources")
    prv_forced = pickle.loads(prv_forced) if prv_forced else []
    for candidate in prv_forced:
        candidate = LSSTForcedPhotometryParser.parse(candidate)
        candidate.update({"aid": aid, "has_stamp": False})
        candidate["extra_fields"].update({"parent_candid": parent})
        candidate.pop("stamps", None)
        forced_photometries.append(candidate)

    alert["extra_fields"]["parent_candid"] = None
    return {
        "aid": alert["aid"],
        "detections": detections,
        "non_detections": [],
        "forced_photometries": forced_photometries,
    }
