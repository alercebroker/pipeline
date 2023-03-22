from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

FID = {
    "u": 0,
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
    "Y": 5,
}


class LSSTParser(SurveyParser):
    _source = "LSST"
    _ignore_in_extra_fields = ["cutoutScience", "cutoutDifference", "cutoutTemplate"]

    _mapping = [
        Mapper("candid", origin="diaSourceId"),
        Mapper("oid", origin="diaObjectId"),
        Mapper("tid", lambda: LSSTParser._source),
        Mapper("pid", lambda: None),
        Mapper("fid", lambda x: FID[x], origin="filterName"),
        Mapper("mjd", origin="midPointTai"),
        Mapper("ra", lambda x: abs(x), origin="ra"),  # TODO: Is it really possible for it to be negative??
        Mapper("dec", origin="decl"),
        # TODO: We're missing e_ra and e_dec. Object meanra, meandec calculations won't work
        Mapper("mag", origin="psFlux"),  # TODO: Are these really magnitudes and not flux?
        Mapper("e_mag", origin="psFluxErr"),  # TODO: Are these really magnitudes and not flux?
        Mapper("isdiffpos", lambda: None),  # TODO: Check if this field can be extracted
    ]

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {
            "cutoutScience": None,
            "cutoutTemplate": message["cutoutTemplate"],
            "cutoutDifference": message["cutoutDifference"],
        }

    @classmethod
    def parse_message(cls, message: dict) -> GenericAlert:
        # the alert data is actually in candidate
        candidate = message["diaSource"].copy()
        # include all top fields
        candidate.update({k: v for k, v in message.items() if k != "diaSource"})
        return super(LSSTParser, cls).parse_message(candidate)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'diaSource' in message.keys()
