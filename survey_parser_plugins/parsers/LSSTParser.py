import math

from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

ERROR = 0.1  # Dummy value for elasticc tests


def _e_ra(dec):
    try:
        return ERROR / abs(math.cos(dec))
    except ZeroDivisionError:
        return float("nan")


class LSSTParser(SurveyParser):
    _source = "LSST"

    _mapping = [
        Mapper("candid", origin="diaSourceId"),
        Mapper("oid", origin="diaObjectId"),
        Mapper("tid", lambda: LSSTParser._source),
        Mapper("sid", lambda: LSSTParser._source),
        Mapper("pid", lambda: -999),  # Using dummy value for elasticc tests
        Mapper("fid", origin="filterName"),
        Mapper("mjd", origin="midPointTai"),
        Mapper("ra", lambda x: x, origin="ra"),
        Mapper("e_ra", lambda y: _e_ra(y), extras=["decl"]),
        Mapper("dec", origin="decl"),
        Mapper("e_dec", lambda: ERROR),
        Mapper("mag", origin="psFlux"),  # TODO: Are these really magnitudes and not flux?
        Mapper("e_mag", origin="psFluxErr"),  # TODO: Are these really magnitudes and not flux?
        Mapper("isdiffpos", lambda x: int(x / abs(x)), extras=["psFlux"]),
    ]

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {
            "science": None,
            "template": message["cutoutTemplate"],
            "difference": message["cutoutDifference"],
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
