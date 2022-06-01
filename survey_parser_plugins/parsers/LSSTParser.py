from ..core import GenericAlert, SurveyParser


class LSSTParser(SurveyParser):
    _source = "LSST"
    _generic_alert_message_key_mapping = {
        "candid": "diaSourceId",
        "mjd": "midPointTai",
        "fid": None,  # This field is modified below in the code
        "rfid": None,
        "isdiffpos": None,
        "pid": None,
        "ra": "ra",
        "dec": "decl",
        "rb": None,
        "rbversion": None,
        "mag": "psFlux",
        "e_mag": "psFluxErr",
    }

    _fid_mapper = {  # u, g, r, i, z, Y
        "u": 0,
        "g": 1,
        "r": 2,
        "i": 3,
        "z": 4,
        "Y": 5,
    }

    @classmethod
    def parse_message(cls, message: dict) -> GenericAlert:
        try:
            oid = message["diaSource"]["diaObjectId"]
            # get stamps
            stamps = {
                "cutoutDifference": message["cutoutDifference"],
                "cutoutTemplate": message["cutoutTemplate"]
            }

            candidate = message["diaSource"]
            generic_alert_message = cls._generic_alert_message(candidate, cls._generic_alert_message_key_mapping)

            # inclusion of extra attributes
            generic_alert_message['oid'] = oid
            generic_alert_message['tid'] = cls._source
            generic_alert_message['fid'] = cls._fid_mapper[candidate["filterName"]]
            # inclusion of stamps
            generic_alert_message["stamps"] = stamps
            # attributes modification
            # ex: jd -> mjd
            # possible attributes
            for k in message.keys():
                if k not in ["diaSource", "cutoutDifference", "cutoutTemplate"]:
                    generic_alert_message["extra_fields"][k] = message[k]
            return GenericAlert(**generic_alert_message)
        except KeyError as e:
            raise KeyError(f"This parser can't parse message: missing {e} key")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'diaSource' in message.keys()
