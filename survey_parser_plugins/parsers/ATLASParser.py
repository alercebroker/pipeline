from survey_parser_plugins.core import GenericAlert, SurveyParser


class ATLASParser(SurveyParser):
    _source = "ATLAS"
    _exclude_keys = ["candid", "mjd", "RA", "Dec", "Mag", "Dmag",
                     "Major", "Minor", "X", "Y"]

    @classmethod
    def parse_message(cls, message: dict, extra_fields: bool = False) -> GenericAlert:
        try:
            oid = message['objectId']
            message = message['candidate'].copy()
            return GenericAlert(
                survey_id=oid,
                survey_name=cls._source,
                candid=message['candid'],
                mjd=message['mjd'],
                fid=-1,
                ra=message['RA'],
                dec=message['Dec'],
                rb=-1,
                mag=message['Mag'],
                sigmag=message['Dmag'],
                aimage=message['Major'],
                bimage=message['Minor'],
                xpos=message['X'],
                ypos=message['Y'],
                extra_fields={
                    k: message[k]
                    for k in message.keys()
                    if k not in cls._exclude_keys
                    } if extra_fields else {}
                )
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def get_source(cls):
        return cls._source

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]
