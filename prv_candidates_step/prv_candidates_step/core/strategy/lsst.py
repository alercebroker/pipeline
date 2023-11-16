import pickle
from dataclasses import asdict

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.parsers import LSSTParser


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
    _mapping = LSSTParser._mapping

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(LSSTForcedPhotometryParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, ra: float, dec: float) -> dict:
        message = message.copy()
        message["diaSourceId"] = message.pop("diaForcedSourceId")
        message["ra"] = ra
        message["decl"] = dec
        return asdict(cls.parse_message(message))


def extract_detections_and_non_detections(alert: dict) -> dict:
    prv_candidates = alert["extra_fields"].pop("prvDiaSources")
    prv_forced = alert["extra_fields"].pop("prvDiaForcedSources")

    detections = [alert.copy()]

    aid, parent = alert["aid"], alert["candid"]
    ra, dec = alert["ra"], alert["dec"]

    if "parent_candid" in alert["extra_fields"]:
        alert["extra_fields"].pop("parent_candid")

    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []
    for candidate in prv_candidates:
        candidate = LSSTPreviousDetectionsParser.parse(candidate)
        candidate.update(
            {
                "aid": aid,
                "has_stamp": False,
                "forced": False,
                "parent_candid": parent,
                "extra_fields": {**alert["extra_fields"], **candidate["extra_fields"]},
            }
        )
        candidate.pop("stamps", None)
        detections.append(candidate)

    prv_forced = pickle.loads(prv_forced) if prv_forced else []
    for candidate in prv_forced:
        candidate = LSSTForcedPhotometryParser.parse(candidate, ra, dec)
        candidate.update(
            {
                "aid": aid,
                "has_stamp": False,
                "forced": True,
                "parent_candid": parent,
                "extra_fields": {**alert["extra_fields"], **candidate["extra_fields"]},
            }
        )
        candidate.pop("stamps", None)
        detections.append(candidate)

    return {"aid": alert["aid"], "detections": detections, "non_detections": []}
