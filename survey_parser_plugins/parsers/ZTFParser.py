from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}


class ZTFParser(SurveyParser):
    _source = "ZTF"

    _mapping = [
        Mapper("candid", origin="candid"),
        Mapper("oid", origin="objectId"),
        Mapper("tid", lambda: ZTFParser._source),
        Mapper("pid", origin="pid"),
        Mapper("fid", origin="fid"),
        Mapper("mjd", lambda x: x - 2400000.5, origin="jd"),
        Mapper("ra", origin="ra"),
        Mapper("e_ra", lambda x, y: x if x else ERRORS[y], origin="sigmara", extras=["fid"], required=False),
        Mapper("dec", origin="dec"),
        Mapper("e_dec", lambda x, y: x if x else ERRORS[y], origin="sigmadec", extras=["fid"], required=False),
        Mapper("mag", origin="magpsf"),
        Mapper("e_mag", origin="sigmapsf"),
        Mapper("isdiffpos", lambda x: 1 if x in ["t", "1"] else -1, origin="isdiffpos"),
    ]

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {
            "cutoutScience": message["cutoutScience"]["stampData"],
            "cutoutTemplate": message["cutoutTemplate"]["stampData"],
            "cutoutDifference": message["cutoutDifference"]["stampData"],
        }

    @classmethod
    def parse_message(cls, message) -> GenericAlert:
        # the alert data is actually in candidate
        candidate = message["candidate"].copy()
        # additional fields from top-level message are included here
        fields_from_top = ["objectId", "prv_candidates", "cutoutScience", "cutoutTemplate", "cutoutDifference"]
        candidate.update({k: v for k, v in message.items() if k in fields_from_top})
        return super(ZTFParser, cls).parse_message(candidate)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return "publisher" in message and cls._source in message["publisher"]
