from survey_parser_plugins.core import SurveyParser


class ATLASParser(SurveyParser):
    _source = "ATLAS"

    @classmethod
    def parse_message(cls, message: dict, extra_fields: bool = False) -> dict:
        return {
            "survey_id": "objectId",
            "candid": "candid",
            "mjd": "mjd",
            # "fid": "fid",
            "ra": 0,
            "dec": 0,
            # "rb": "rb",
            "mag": "Mag",
            "sigmag": "Dmag",
            "aimage": "Major",
            "bimage": "Minor",
        }

    @classmethod
    def get_source(cls):
        return cls._source

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
