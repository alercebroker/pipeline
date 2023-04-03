from typing import List, Union
from dataclasses import asdict
from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.parsers.ZTFParser import ERRORS
from survey_parser_plugins.core.mapper import Mapper


# Implementation a new parser for PreviousCandidates with SurveyParser signs.
class ZTFPreviousDetectionsParser(SurveyParser):
    _source = "ZTF"
    _mapping = [
        Mapper("oid", lambda: ZTFPreviousDetectionsParser._oid),
        Mapper("tid", lambda: ZTFPreviousDetectionsParser._source),
        Mapper("candid", origin="candid"),
        Mapper("mjd", lambda x: x - 2400000.5, origin="jd"),
        Mapper("fid", origin="fid"),
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
        return {"science": None, "template": None, "difference": None}

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, messages: List[dict], oid: str, aid: str, parent_candid: Union[str, int]) -> List[dict]:
        def parse_to_dict(message: dict) -> dict:
            alert = asdict(cls.parse_message(message))
            alert["aid"] = aid
            alert["extra_fields"]["parent_candid"] = parent_candid
            alert["has_stamp"] = False
            return alert

        cls.set_oid(oid)
        return list(map(parse_to_dict, messages))
