from ..core import GenericAlert, SurveyParser


class ATLASParser(SurveyParser):
    _source = "ATLAS"
    _generic_alert_message_key_mapping = {
        "candid": "candid",
        "mjd": "mjd",
        "fid": None, # This field is modified below in the code
        "rfid": None,
        "isdiffpos": None,
        "pid": None,
        "ra": "RA",
        "dec": "Dec",
        "rb": None,
        "rbversion": None,
        "mag": "Mag",
        "sigmag": "Dmag",
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
            generic_alert_message['tid'] = message["publisher"]
            generic_alert_message['aid'] = oid
            generic_alert_message['fid'] = cls._fid_mapper[candidate["filter"]]
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            # possible attributes
            sigmaradec = 0.07
            generic_alert_message["sigmara"] = candidate["sigmara"] if "sigmara" in candidate else sigmaradec 
            generic_alert_message["sigmadec"] = candidate["sigmadec"] if "sigmadec" in candidate else sigmaradec 
            return GenericAlert(**generic_alert_message)
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
