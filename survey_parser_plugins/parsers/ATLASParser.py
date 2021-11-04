from ..core import GenericAlert, SurveyParser
import numpy as np


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
        "e_mag": "Dmag",
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
            generic_alert_message['tid'] = cls._source # This must be more exact (for telescopes)
            generic_alert_message['aid'] = oid
            generic_alert_message['fid'] = cls._fid_mapper[candidate["filter"]]
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            # possible attributes
            e_dec = 0.14
            e_ra = 0.14/abs(np.cos(generic_alert_message["dec"]))
            generic_alert_message["e_ra"] = candidate["sigmara"] if "sigmara" in candidate else e_ra
            generic_alert_message["e_dec"] = candidate["sigmadec"] if "sigmadec" in candidate else e_dec 
            return GenericAlert(**generic_alert_message)
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
