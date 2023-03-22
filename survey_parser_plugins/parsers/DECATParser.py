from typing import List

from ..core import GenericAlert, SurveyParser
from ..core.mapper import Mapper

FID = {
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
}


class DECATParser(SurveyParser):
    _source = "DECAT"

    _mapping = [
        Mapper("candid", origin="sourceid"),
        Mapper("oid", origin="objectid"),
        Mapper("tid", lambda: DECATParser._source),
        Mapper("pid", lambda: ""),
        Mapper("fid", lambda x: FID[x[0]], origin="filter"),
        Mapper("mjd", origin="mjd"),
        Mapper("ra", origin="ra"),
        Mapper("dec", origin="dec"),
        Mapper("mag", origin="mag"),
        Mapper("e_mag", origin="magerr"),
        Mapper("isdiffpos", lambda: 1),
    ]

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return {}

    @classmethod
    def parse_message(cls, message) -> List[GenericAlert]:
        candidates = [{"objectid": message["objectid"], **msg} for msg in message["sources"]]
        return [super(DECATParser, cls).parse_message(cand) for cand in candidates]

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        if "objectid" in message.keys() and "sources" in message.keys():
            return "DC" in message["objectid"] and "DECam" in message["sources"][0]["filter"]
        return False
