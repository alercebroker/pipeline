from ..core import GenericAlert, SurveyParser


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
    def parse_message(cls, message) -> GenericAlert:
        try:
            oid = message["objectId"]
            # get stamps
            stamps = {
                "cutoutScience": message["cutoutScience"]["stampData"],
                "cutoutDifference": message["cutoutDifference"]["stampData"]
            }

            candidate = message["candidate"]
            generic_alert_message = cls._generic_alert_message(candidate, cls._generic_alert_message_key_mapping)

            # inclusion of extra attributes
            generic_alert_message['oid'] = oid
            generic_alert_message['sid'] = cls._source
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            return GenericAlert(**generic_alert_message)
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
