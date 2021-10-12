from survey_parser_plugins.core import SurveyParser

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
    def _get_filter(self, fid) -> int:
        fid = fid[0]
        return MAP_FID[fid]

    @classmethod
    def parse_message(cls, message, extra_fields=False):
        oid = message["objectid"]
        message = message["sources"].copy()
        return [{
            "survey_id": oid,
            "candid": msg['sourceid'],
            "mjd": msg['mjd'],
            "fid": cls._get_filter(msg['filter']),
            "ra": float(msg['ra']),
            "dec": float(msg['dec']),
            "rb": msg['rb'],
            "mag": msg['mag'],
            "sigmag": msg["magerr"],
            "aimage": None,
            "bimage": None,
            "extra_fields": {
                k: msg[k]
                for k in msg.keys()
                if k not in cls._exclude_keys
            } if extra_fields else None
        }
            for msg in message
        ]

    @classmethod
    def get_source(cls):
        return cls._source

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        if "objectid" in message.keys() and "sources" in message.keys():
            return "DC" in message["objectid"] and "DECam" in message["sources"][0]["filter"]
        return False
