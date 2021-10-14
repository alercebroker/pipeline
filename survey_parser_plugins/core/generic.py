import abc
from dataclasses import dataclass, field


@dataclass
class GenericAlert:
    """
    Class for keeping track of an alert of astronomical surveys.
    """
    oid: str  # name of object (name from survey)
    sid: str  # identifier of survey (name of survey)
    candid: int
    mjd: float
    fid: int
    ra: float
    dec: float
    rb: float
    mag: float
    sigmag: float
    aimage: float or None
    bimage: float or None
    xpos: float or None
    ypos: float or None
    aid: int = None  # alerce id of the object
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return self.__getattribute__(item)


class SurveyParser(abc.ABC):
    _source = None
    _generic_alert_message_key_mapping = {}

    @abc.abstractmethod
    def parse_message(self, message: GenericAlert) -> GenericAlert:
        """
        :param message: A single message from an astronomical survey. Typically this corresponds a dict.

        Note that the Creator may also provide some default implementation of the factory method.
        """
    @abc.abstractmethod
    def can_parse(self, message: dict) -> bool:
        """
        :param message: message of any survey.
        Note that the Creator may also provide some default implementation of the factory method.
        """

    @classmethod
    def _generic_alert_message(cls, message: dict, generic_alert_message_key_mapping: dict) -> dict:
        """
        Create the generic content from any survey. Also add `extra_fields` key with all data that isn't generic key.

        :param message: message of any survey.
        :param generic_alert_message_key_mapping: dictionary with the mapping from GenericAlert to survey schema.
        for example {"generic_key": "survey_key"}. If the survey hasn't some key from generic schema, put None.
        :return:
        """
        generic_alert_message = {}
        for key, value in generic_alert_message_key_mapping.items():
            if value is None:
                generic_alert_message[key] = None
            else:
                generic_alert_message[key] = message[value]

        generic_alert_message["extra_fields"] = {
            k: message[k]
            for k in message.keys()
            if k not in generic_alert_message_key_mapping.values()
        }
        return generic_alert_message

    @classmethod
    def get_source(cls) -> str:
        """

        :return: source of the parser.
        """
        return cls._source

    @classmethod
    def get_key_mapping(cls) -> dict:
        """

        :return: dictionary for map raw message to generic alerts.
        """
        return cls._generic_alert_message_key_mapping
