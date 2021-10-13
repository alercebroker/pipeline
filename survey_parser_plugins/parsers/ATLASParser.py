from survey_parser_plugins.core import GenericAlert, SurveyParser


class ATLASParser(SurveyParser):
    _source = "ATLAS"
    _generic_alert_message_key_mapping = {
        "candid": "candid",
        "mjd": "mjd",
        "fid": None,
        "ra": "RA",
        "dec": "Dec",
        "rb": None,
        "mag": "Mag",
        "sigmag": "Dmag",
        "aimage": "Major",
        "bimage": "Minor",
        "xpos": "X",
        "ypos": "Y",
    }

    @classmethod
    def parse_message(cls, message: dict) -> GenericAlert:
        try:
            oid = message["objectId"]
            message = message["candidate"].copy()
            generic_alert_message = cls._generic_alert_message(
                message, cls._generic_alert_message_key_mapping)
            # inclusion of extra attributes
            generic_alert_message['survey_id'] = oid
            generic_alert_message['survey_name'] = cls._source
            return GenericAlert(**generic_alert_message)
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def get_source(cls):
        return cls._source

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
