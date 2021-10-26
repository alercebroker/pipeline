from ..core import GenericAlert, SurveyParser
from typing import List

MAP_FID = {
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
}


class DECATParser(SurveyParser):
    _source = "DECAT"
    _exclude_keys = []

    @classmethod
    def _get_filter(cls, fid) -> int:
        fid = fid[0]
        return MAP_FID[fid]

    @classmethod
    def parse_message(cls, message) -> List[GenericAlert]:
        try:
            oid = message["objectid"]
            message = message["sources"].copy()
            return [GenericAlert(
                oid=oid,
                tid=cls._source,
                pid="",
                candid=msg['sourceid'],
                mjd=msg['mjd'],
                fid=cls._get_filter(msg['filter']),
                ra=msg['ra'],
                dec=msg['dec'],
                rb=msg['rb'],
                mag=msg['mag'],
                sigmag=msg["magerr"],
                )
                for msg in message
            ]
        except KeyError:
            raise KeyError("This parser can't parse message")


    @classmethod
    def can_parse(cls, message: dict) -> bool:
        if "objectid" in message.keys() and "sources" in message.keys():
            return "DC" in message["objectid"] and "DECam" in message["sources"][0]["filter"]
        return False
