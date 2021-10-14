from survey_parser_plugins.core import GenericAlert, SurveyParser


class ZTFParser(SurveyParser):
    _source = "ZTF"
    _generic_alert_message_key_mapping = {
        "candid": "candid",
        "mjd": "jd",
        "fid": "fid",
        "ra": "ra",
        "dec": "dec",
        "rb": "rb",
        "mag": "magpsf",
        "sigmag": "sigmapsf",
        "aimage": "aimage",
        "bimage": "bimage",
        "xpos": "xpos",
        "ypos": "ypos",
    }

    @classmethod
    def parse_message(cls, message) -> GenericAlert:
        try:
            oid = message["objectId"]
            prv_candidates = message["prv_candidates"]
            # get stamps
            stamps = {
                "cutoutScience": message["cutoutScience"]["stampData"],
                "cutoutTemplate": message["cutoutTemplate"]["stampData"],
                "cutoutDifference": message["cutoutDifference"]["stampData"]
            }

            candidate = message["candidate"]
            generic_alert_message = cls._generic_alert_message(candidate, cls._generic_alert_message_key_mapping)

            # inclusion of extra attributes
            generic_alert_message['oid'] = oid
            generic_alert_message['sid'] = cls._source
            generic_alert_message["extra_fields"]["prv_candidates"] = prv_candidates
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            generic_alert_message['mjd'] = generic_alert_message['mjd'] - 2400000.5
            return GenericAlert(**generic_alert_message)
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message[
            "publisher"]
