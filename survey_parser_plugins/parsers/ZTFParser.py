from survey_parser_plugins.core import SurveyParser


class ZTFParser(SurveyParser):
    _source = "ZTF"

    @classmethod
    def parse_message(cls, message):
        oid = message["objectId"]
        message = message["candidate"].copy()
        return {
            "alerce_id": oid,
            "survey_id": oid,
            "candid": message['candid'],
            "mjd": message['jd'] - 2400000.5,
            "fid": message['fid'],
            "ra": message['ra'],
            "dec": message['dec'],
            "rb": message['rb'],
            "mag": message['magpsf'],
            "sigmag": message["sigmapsf"],
            "aimage": message["aimage"],
            "bimage": message["bimage"],
        }

    @classmethod
    def get_source(cls):
        return cls._source
