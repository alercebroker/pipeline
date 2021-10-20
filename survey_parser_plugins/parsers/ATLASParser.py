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

    _fid_mapper = {
        "g": 1,
        "r": 2,
        "i": 3,
        "c": 4,
        "o": 5,
        "H": 6
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
            generic_alert_message['aid'] = oid
            if "filter" in candidate.keys():
                generic_alert_message['fid'] = cls._fid_mapper[candidate["filter"]]
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            return GenericAlert(**generic_alert_message)
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
