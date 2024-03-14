import copy
import pickle
from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.parsers import LSSTParser
from survey_parser_plugins.core.mapper import Mapper


def lsst_prv_mapping():
    mapping = copy.deepcopy(LSSTParser._mapping)
    mapping.update({"candid": Mapper(str, origin="diaSourceId")})
    return mapping


class LSSTPreviousDetectionsParser(SurveyParser):
    _mapping = lsst_prv_mapping()

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(LSSTPreviousDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict) -> dict:
        message = message.copy()
        message["alertId"] = str(message["diaSourceId"])
        return cls.parse_message(message).to_dict()


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
        message["alertId"] = str(message.pop("diaForcedSourceId"))
        message["ra"] = ra
        message["decl"] = dec
        return cls.parse_message(message).to_dict()


def extract_detections_and_non_detections(alert: dict) -> dict:
    prv_candidates = alert["extra_fields"].pop("prvDiaSources")
    prv_forced = alert["extra_fields"].pop("prvDiaForcedSources")

    detections = [alert.copy()]

    oid, aid, parent = alert["oid"], alert["aid"], alert["candid"]
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

    return {"oid": alert["oid"], "detections": detections, "non_detections": []}
