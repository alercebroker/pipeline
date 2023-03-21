from survey_parser_plugins.core import SurveyParser
from typing import List, Union


# Implementation a new parser for PreviousCandidates with SurveyParser signs.
class ZTFPreviousDetectionsParser(SurveyParser):
    _source = "ZTF"
    _celestial_errors = {
        1: 0.065,
        2: 0.085,
        3: 0.01,
    }
    _generic_alert_message_key_mapping = {
        "candid": "candid",
        "mjd": "jd",
        "fid": "fid",
        "pid": "pid",
        "ra": "ra",
        "dec": "dec",
        "mag": "magpsf",
        "e_mag": "sigmapsf",
        "isdiffpos": "isdiffpos",
        "rb": "rb",
        "rbversion": "rbversion",
    }

    @classmethod
    def parse_message(
        cls, message: dict, oid: str, aid: str, parent_candid: Union[str, int]
    ) -> dict:
        if not cls.can_parse(message):
            raise KeyError("This parser can't parse message")
        prv_content = cls._generic_alert_message(
            message, cls._generic_alert_message_key_mapping
        )
        # inclusion of extra attributes
        prv_content["oid"] = oid
        # prv_content["aid"] = id_generator(prv_content["ra"], prv_content["dec"])
        prv_content["tid"] = cls._source
        # attributes modification
        prv_content["aid"] = aid
        prv_content["mjd"] = prv_content["mjd"] - 2400000.5
        prv_content["isdiffpos"] = 1 if prv_content["isdiffpos"] in ["t", "1"] else -1
        prv_content["extra_fields"]["parent_candid"] = parent_candid
        e_radec = cls._celestial_errors[prv_content["fid"]]
        prv_content["e_ra"] = (
            prv_content["sigmara"] if "sigmara" in prv_content else e_radec
        )
        prv_content["e_dec"] = (
            prv_content["sigmadec"] if "sigmadec" in prv_content else e_radec
        )
        prv_content["stamps"] = {"science": None, "difference": None, "template": None}
        return prv_content

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(
        cls, messages: List[dict], oid: str, aid: str, parent_candid: Union[str, int]
    ) -> List[dict]:
        def curry_parse(message: dict):
            return cls.parse_message(message, oid, aid, parent_candid)

        return list(map(curry_parse, messages))
