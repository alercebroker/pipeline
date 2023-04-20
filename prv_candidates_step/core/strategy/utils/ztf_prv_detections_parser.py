from typing import List, Union
from dataclasses import asdict
from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.parsers.ZTFParser import ERRORS, FILTER
from survey_parser_plugins.core.mapper import Mapper


# Implementation a new parser for PreviousCandidates with SurveyParser signs.
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
    def parse(
        cls, messages: List[dict], oid: str, aid: str, parent: Union[str, int]
    ) -> List[dict]:
        return [cls.parse_to_dict(message, aid, oid, parent) for message in messages]

    @classmethod
    def parse_to_dict(cls, message: dict, aid, oid, parent) -> dict:
        cls.set_oid(oid)
        alert = asdict(cls.parse_message(message))
        alert["aid"] = aid
        alert["extra_fields"]["parent_candid"] = parent
        alert["has_stamp"] = False
        alert.pop("stamps")
        return alert
