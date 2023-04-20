import math

from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

ERROR = 0.1  # UPDATE ME: Dummy value for elasticc tests


def _e_ra(dec):
    try:
        return ERROR / abs(math.cos(dec))
    except ZeroDivisionError:
        return float("nan")


class LSSTParser(SurveyParser):
    _source = "LSST"

    _mapping = {
        "candid": Mapper(origin="diaSourceId"),
        "oid": Mapper(origin="diaObjectId"),
        "tid": Mapper(lambda: LSSTParser._source),
        "sid": Mapper(lambda: LSSTParser._source),
        "pid": Mapper(lambda: -999),  # UPDATE ME
        "fid": Mapper(origin="filterName"),
        "mjd": Mapper(origin="midPointTai"),
        "ra": Mapper(lambda x: x, origin="ra"),
        "e_ra": Mapper(lambda y: _e_ra(y), extras=["decl"]),  # UPDATE ME
        "dec": Mapper(origin="decl"),
        "e_dec": Mapper(lambda: ERROR),  # UPDATE ME
        "mag": Mapper(origin="psFlux"),  # TODO: Are these really magnitudes and not flux?
        "e_mag": Mapper(origin="psFluxErr"),  # TODO: Are these really magnitudes and not flux?
        "isdiffpos": Mapper(lambda x: int(x / abs(x)), extras=["psFlux"]),
    }

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
