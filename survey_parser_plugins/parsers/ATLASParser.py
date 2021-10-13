from survey_parser_plugins.core import GenericAlert, SurveyParser


class ATLASParser(SurveyParser):
    _source = "ATLAS"

    @classmethod
    def parse_message(cls, message: dict, extra_fields: bool = False) -> GenericAlert:
        return GenericAlert(
            "",  # surveyid
            cls._source,
            0,  # candid
            0,  # mjd
            0,  # fid
            0,  # ra
            0,  # dec
            0,  # rb
            0,  # mag
            0,  # sigmag
            0,
            0,
            0,
            0,
        )

    @classmethod
    def get_source(cls):
        return cls._source

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
