from ..core import GenericAlert, SurveyParser


class ZTFParser(SurveyParser):
    _source = "ZTF"
    _celestial_errors = {
        1: 0.065,
        2: 0.085,
        3: 0.01,
    }
    _mapping = {
        "candid": "candid",
        "mjd": "jd",
        "fid": "fid",
        "pid": "pid",
        "rfid": "rfid",
        "ra": "ra",
        "dec": "dec",
        "mag": "magpsf",
        "e_mag": "sigmapsf",
        "isdiffpos": "isdiffpos",
        "rb": "rb",
        "rbversion": "rbversion"
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
            generic_alert_message = cls._generic_alert_message(candidate)

            # inclusion of extra attributes
            generic_alert_message['oid'] = oid
            generic_alert_message['tid'] = cls._source
            generic_alert_message["extra_fields"]["prv_candidates"] = prv_candidates
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            generic_alert_message["isdiffpos"] = 1 if generic_alert_message["isdiffpos"] in ["t", "1"] else -1
            generic_alert_message['mjd'] = generic_alert_message['mjd'] - 2400000.5

            # possible attributes
            e_radec = cls._celestial_errors[candidate["fid"]]
            generic_alert_message["e_ra"] = candidate["sigmara"] if "sigmara" in candidate else e_radec 
            generic_alert_message["e_dec"] = candidate["sigmadec"] if "sigmadec" in candidate else e_radec 
            return GenericAlert(**generic_alert_message)
        except KeyError as e:
            raise KeyError(f"This parser can't parse message: missing {e} key")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
