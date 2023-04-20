import pickle
from dataclasses import asdict

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.core.mapper import Mapper
from survey_parser_plugins.parsers.ZTFParser import FILTER, ERRORS


class ZTFPreviousDetectionsParser(SurveyParser):
    _oid = ""
    _source = "ZTF"
    _mapping = [
        Mapper("oid", lambda: ZTFPreviousDetectionsParser._oid),
        Mapper("sid", lambda: ZTFPreviousDetectionsParser._source),
        Mapper("tid", lambda: ZTFPreviousDetectionsParser._source),
        Mapper("candid", origin="candid"),
        Mapper("mjd", lambda x: x - 2400000.5, origin="jd"),
        Mapper("fid", lambda x: FILTER[x], origin="fid"),
        Mapper("pid", origin="pid"),
        Mapper("ra", origin="ra"),
        Mapper(
            "e_ra",
            lambda x, y: x if x else ERRORS[y],
            origin="sigmara",
            extras=["fid"],
            required=False,
        ),
        Mapper("dec", origin="dec"),
        Mapper(
            "e_dec",
            lambda x, y: x if x else ERRORS[y],
            origin="sigmadec",
            extras=["fid"],
            required=False,
        ),
        Mapper("mag", origin="magpsf"),
        Mapper("e_mag", origin="sigmapsf"),
        Mapper("isdiffpos", lambda x: 1 if x in ["t", "1"] else -1, origin="isdiffpos"),
    ]

    @classmethod
    def set_oid(cls, oid: str):
        cls._oid = oid

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFPreviousDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse_to_dict(cls, message: dict, aid, oid, parent) -> dict:
        cls.set_oid(oid)
        alert = asdict(cls.parse_message(message))
        alert["aid"] = aid
        alert["extra_fields"]["parent_candid"] = parent
        alert["has_stamp"] = False
        alert.pop("stamps")
        return alert


class ZTFNonDetectionsParser:
    @classmethod
    def parse_non_detection(cls, non_detection: dict, aid: str, oid: str):
        return {
            "aid": aid,
            "tid": "ZTF",
            "oid": oid,
            "sid": "ZTF",
            "mjd": cls.convert_mjd(non_detection["jd"]),
            "fid": FILTER[non_detection["fid"]],
            "diffmaglim": non_detection["diffmaglim"],
        }

    @staticmethod
    def convert_mjd(jd: float):
        return jd - 2400000.5


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
