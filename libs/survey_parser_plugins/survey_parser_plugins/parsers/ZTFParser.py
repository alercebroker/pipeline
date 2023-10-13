import math
from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}

FILTER = {
    1: "g",
    2: "r",
    3: "i",
}


def _e_ra(dec, fid):
    try:
        return ERRORS[fid] / abs(math.cos(math.radians(dec)))
    except ZeroDivisionError:
        return float("nan")


class ZTFParser(SurveyParser):
    _source = "ZTF"

    _mapping = {
        "candid": Mapper(str, origin="candid"),
        "oid": Mapper(str, origin="objectId"),
        "tid": Mapper(lambda: ZTFParser._source),
        "sid": Mapper(lambda: ZTFParser._source),
        "pid": Mapper(origin="pid"),
        "fid": Mapper(lambda x: FILTER[x], origin="fid"),
        "mjd": Mapper(lambda x: x - 2400000.5, origin="jd"),
        "ra": Mapper(origin="ra"),
        "e_ra": Mapper(
            lambda x, y, z: x if x else _e_ra(y, z),
            origin="sigmara",
            extras=["dec", "fid"],
            required=False,
        ),
        "dec": Mapper(origin="dec"),
        "e_dec": Mapper(
            lambda x, y: x if x else ERRORS[y],
            origin="sigmadec",
            extras=["fid"],
            required=False,
        ),
        "mag": Mapper(origin="magpsf"),
        "e_mag": Mapper(origin="sigmapsf"),
        "isdiffpos": Mapper(lambda x: 1 if x in ["t", "1"] else -1, origin="isdiffpos"),
    }

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {
            "science": message["cutoutScience"]["stampData"],
            "template": message["cutoutTemplate"]["stampData"],
            "difference": message["cutoutDifference"]["stampData"],
        }

    @classmethod
    def parse_message(cls, message) -> GenericAlert:
        # the alert data is actually in candidate
        candidate = message["candidate"].copy()
        # additional fields from top-level message are included here
        fields_from_top = [
            "objectId",
            "prv_candidates",
            "cutoutScience",
            "cutoutTemplate",
            "cutoutDifference",
            "brokerIngestTimestamp",
            "surveyPublishTimestamp",
            "fp_hists",
        ]
        candidate.update({k: v for k, v in message.items() if k in fields_from_top})
        return super(ZTFParser, cls).parse_message(candidate)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return "publisher" in message and cls._source in message["publisher"]
