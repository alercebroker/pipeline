import abc
import functools
from dataclasses import dataclass, field
from typing import Sequence, Set

from .mapper import Mapper


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
    mag: float
    e_mag: float
    isdiffpos: int
    e_ra: float = None
    e_dec: float = None
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return self.__getattribute__(item)


class SurveyParser(abc.ABC):
    _source: str
    _mapping: Sequence[Mapper]
    _from_top: Sequence[str] = []

    @classmethod
    @functools.lru_cache
    def _ignore_for_extra_fields(cls) -> Set[str]:
        ignore = {mapper.origin for mapper in cls._mapping if mapper.origin is not None}
        ignore.update(cls._from_top)
        return ignore

    @classmethod
    @abc.abstractmethod
    def _extract_stamps(cls, message: dict) -> dict:
        pass

    @classmethod
    def parse_message(cls, message: dict) -> GenericAlert:
        """
        :param message: A single message from an astronomical survey

        Note that the Creator may also provide some default implementation of the factory method.
        """
        generic = {mapper.field: mapper(message) for mapper in cls._mapping}

        stamps = cls._extract_stamps(message)
        extra_fields = {k: v for k, v in message.items() if k not in cls._ignore_for_extra_fields()}
        return GenericAlert(**generic, stamps=stamps, extra_fields=extra_fields)

    @abc.abstractmethod
    def can_parse(self, message: dict) -> bool:
        """
        :param message: message of any survey.
        Note that the Creator may also provide some default implementation of the factory method.
        """

    @classmethod
    def get_source(cls) -> str:
        """

        :return: source of the parser.
        """
        return cls._source
