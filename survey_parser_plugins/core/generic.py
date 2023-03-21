import abc
from dataclasses import dataclass, field
from typing import Union


@dataclass
class GenericAlert:
    """
    Class for keeping track of an alert of astronomical surveys.
    """
    oid: str  # name of object (name from survey)
    tid: str  # identifier of survey (name of survey)
    pid: int
    candid: str
    mjd: float
    fid: int
    ra: float
    dec: float
    rb: float
    rbversion: str
    mag: float
    e_mag: float
    rfid: int
    isdiffpos: int 
    e_ra: float
    e_dec: float
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return self.__getattribute__(item)


class SurveyParser(abc.ABC):
    _source = None
    _mapping = {}

    @abc.abstractmethod
    def parse_message(self, message: dict) -> GenericAlert:
        """
        :param message: A single message from an astronomical survey

        Note that the Creator may also provide some default implementation of the factory method.
        """
    @abc.abstractmethod
    def can_parse(self, message: dict) -> bool:
        """
        :param message: message of any survey.
        Note that the Creator may also provide some default implementation of the factory method.
        """

    @classmethod
    def _generic_alert_message(cls, message: dict) -> dict:
        """
        Create the generic content from any survey. Also add `extra_fields` key with all data that isn't generic key.

        :param message: message of any survey.
        """
        generic_alert_message = {}
        for key, value in cls._mapping.items():
            if value is None:
                generic_alert_message[key] = None
            else:
                generic_alert_message[key] = message[value]

        generic_alert_message["extra_fields"] = {
            k: message[k]
            for k in message.keys()
            if k not in cls._mapping.values()
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
        return cls._mapping
