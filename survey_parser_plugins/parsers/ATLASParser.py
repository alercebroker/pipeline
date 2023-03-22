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


def _e_ra(dec):
    return ERROR / abs(math.cos(dec))


class ATLASParser(SurveyParser):
    _source = "ATLAS"
    _ignore_in_extra_fields = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]

    _mapping = [
        Mapper("candid", origin="candid"),
        Mapper("oid", origin="objectId"),
        Mapper("tid", origin="publisher"),
        Mapper("pid", origin="pid"),
        Mapper("fid", lambda x: FID[x], origin="filter"),
        Mapper("mjd", origin="mjd"),
        Mapper("ra", origin="RA"),
        Mapper("e_ra", lambda x, y: x if x else _e_ra(y), origin="sigmara", extras=["Dec"], required=False),
        Mapper("dec", origin="Dec"),
        Mapper("e_dec", lambda x: x if x else ERROR, origin="sigmadec", required=False),
        Mapper("mag", origin="Mag"),
        Mapper("e_mag", origin="Dmag"),
        Mapper("isdiffpos", lambda x: 1 if x in ["t", "1"] else -1, origin="isdiffpos"),
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
        # the alert data is actually in candidate
        candidate = message["candidate"].copy()
        # additional fields from top-level message are included here
        fields_from_top = ["objectId", "publisher", "cutoutScience", "cutoutTemplate", "cutoutDifference"]
        candidate.update({k: v for k, v in message.items() if k in fields_from_top})
        return super(ATLASParser, cls).parse_message(candidate)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return "publisher" in message and cls._source in message["publisher"]
