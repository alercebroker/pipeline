import math

from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

ERROR = 0.14


def _e_ra(dec):
    try:
        return ERROR / abs(math.cos(math.radians(dec)))
    except ZeroDivisionError:
        return float("nan")


class ATLASParser(SurveyParser):
    _source = "ATLAS"

    _mapping = {
        "candid": Mapper(str, origin="candid"),
        "oid": Mapper(str, origin="objectId"),
        "tid": Mapper(origin="publisher"),
        "sid": Mapper(lambda: ATLASParser._source),
        "pid": Mapper(origin="pid"),
        "fid": Mapper(origin="filter"),
        "mjd": Mapper(origin="mjd"),
        "ra": Mapper(origin="RA"),
        "e_ra": Mapper(
            lambda x, y: x if x else _e_ra(y),
            origin="sigmara",
            extras=["Dec"],
            required=False,
        ),
        "dec": Mapper(origin="Dec"),
        "e_dec": Mapper(lambda x: x if x else ERROR, origin="sigmadec", required=False),
        "mag": Mapper(origin="Mag"),
        "e_mag": Mapper(origin="Dmag"),
        "isdiffpos": Mapper(lambda x: 1 if x in ["t", "1"] else -1, origin="isdiffpos"),
    }

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {
            "science": message.get("cutoutScience", {}).get("stampData", None),
            "template": None,
            "difference": message.get("cutoutDifference", {}).get("stampData", None),
        }

    @classmethod
    def parse_message(cls, message) -> GenericAlert:
        # the alert data is actually in candidate
        candidate = message["candidate"].copy()
        # additional fields from top-level message are included here
        fields_from_top = [
            "objectId",
            "publisher",
            "cutoutScience",
            "cutoutTemplate",
            "cutoutDifference",
            "brokerIngestTimestamp",
            "surveyPublishTimestamp",
        ]
        candidate.update({k: v for k, v in message.items() if k in fields_from_top})
        return super(ATLASParser, cls).parse_message(candidate)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return "publisher" in message and cls._source in message["publisher"]
