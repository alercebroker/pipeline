import math

from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper


FID = {
    "g": 1,
    "r": 2,
    "i": 3,
    "c": 4,
    "o": 5,
    "H": 6,
}

ERROR = 0.14


class ATLASParser(SurveyParser):
    _source = "ATLAS"
    _from_top = ["objectId", "publisher", "cutoutScience", "cutoutTemplate", "cutoutDifference"]

    _mapping = [
        Mapper("candid", origin="candid"),
        Mapper("oid", origin="objectId"),
        Mapper("tid", origin="publisher"),
        Mapper("pid", origin="pid"),
        Mapper("fid", lambda m, f: FID[m[f]], origin="filter"),
        Mapper("mjd", origin="mjd"),
        Mapper("ra", origin="RA"),
        Mapper("e_ra", lambda m, f: m[f] if f in m else ERROR/abs(math.cos(m["Dec"])), origin="sigmara"),
        Mapper("dec", origin="Dec"),
        Mapper("e_dec", lambda m, f: m[f] if f in m else ERROR, origin="sigmadec"),
        Mapper("mag", origin="Mag"),
        Mapper("e_mag", origin="Dmag"),
        Mapper("isdiffpos", lambda m, f: 1 if m[f] in ["t", "1"] else -1, origin="isdiffpos")
    ]

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {
            "cutoutScience": message["cutoutScience"]["stampData"],
            "cutoutTemplate": None,
            "cutoutDifference": message["cutoutDifference"]["stampData"],
        }

    @classmethod
    def parse_message(cls, message) -> GenericAlert:
        candidate = message["candidate"].copy()
        candidate.update({k: v for k, v in message.items() if k in cls._from_top})
        return super(ATLASParser, cls).parse_message(candidate)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message and cls._source in message["publisher"]
