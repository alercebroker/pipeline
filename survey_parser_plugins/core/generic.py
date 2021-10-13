import abc
from dataclasses import dataclass, field


@dataclass
class GenericAlert:
    """
    Class for keeping track of an alert of astronomical surveys.
    """
    survey_id: str
    survey_name: str
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
    alerce_id: int = None
    extra_fields: dict = field(default_factory=dict)


class SurveyParser(abc.ABC):
    _source = None

    @abc.abstractmethod
    def parse_message(self, message: GenericAlert) -> GenericAlert:
        """
        :param message: A single message from an astronomical survey. Typically this corresponds a dict.

        Note that the Creator may also provide some default implementation of the factory method.
        """

    @abc.abstractmethod
    def get_source(self) -> str:
        """
        Note that the Creator may also provide some default implementation of the factory method.
        """

    @abc.abstractmethod
    def can_parse(self, message: dict) -> bool:
        """
        Note that the Creator may also provide some default implementation of the factory method.
        """

    @classmethod
    def _generic_alert_message(cls, message: dict, generic_alert_message_key_mapping: dict) -> dict:
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
