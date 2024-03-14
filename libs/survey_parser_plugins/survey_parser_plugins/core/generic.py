import abc
import functools
from dataclasses import dataclass, field
from typing import Dict, Sequence, Set

from .mapper import Mapper


@dataclass
class GenericAlert:
    """Alert of astronomical surveys."""

    oid: str  # name of object (from survey)
    tid: str  # telescope identifier
    sid: str  # survey identifier
    pid: int  # processing identifier for image
    candid: str  # candidate identifier (from survey)
    mjd: float  # modified Julian date
    fid: str  # filter identifier
    ra: float  # right ascension
    dec: float  # declination
    mag: float  # difference magnitude
    e_mag: float  # difference magnitude uncertainty
    isdiffpos: int  # sign of the flux difference
    e_ra: float  # right ascension uncertainty
    e_dec: float  # declination uncertainty
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def to_dict(self):
        generic_alert_dict = {
            "oid": self.oid,
            "tid": self.tid,
            "sid": self.sid,
            "pid": self.pid,
            "candid": self.candid,
            "mjd": self.mjd,
            "fid": self.fid,
            "ra": self.ra,
            "dec": self.dec,
            "mag": self.mag,
            "e_mag": self.e_mag,
            "isdiffpos": self.isdiffpos,
            "e_ra": self.e_ra,
            "e_dec": self.e_dec,
            "extra_fields": self.extra_fields,
            "stamps": self.stamps,
        }
        return generic_alert_dict


@dataclass
class GenericNonDetection:
    """Non detection of astronomical surveys."""

    oid: str  # name of object (from survey)
    tid: str  # telescope identifier
    sid: str  # survey identifier
    mjd: float  # modified Julian date
    fid: str  # filter identifier
    diffmaglim: float  # sign of the flux difference
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def to_dict(self):
        generic_non_detection_dict = {
            "oid": self.oid,
            "tid": self.tid,
            "sid": self.sid,
            "mjd": self.mjd,
            "fid": self.fid,
            "diffmaglim": self.diffmaglim,
            "extra_fields": self.extra_fields,
            "stamps": self.stamps,
        }

        return generic_non_detection_dict


class SurveyParser(abc.ABC):
    """Base class for survey parsing. Subclasses are intended to be static.

    The field `_source` is an identifier for the survey.

    The field `_mapping` should have a list of `Mapper` objects and have one entry per field in `GenericAlert`,
    except for `stamps` and `extra_fields`, which are generated through specialized methods.

    The field `_ignore_in_extra_fields` should include fields from the message that are not added to `extra_fields`.
    In addition to the fields described above, the parser automatically ignores all the `origin` fields from the
    `Mapper` objects described in `_mapping`.
    """

    _source: str
    _mapping: Dict[str, Mapper]
    _ignore_in_extra_fields: Sequence[str] = [
        "cutoutScience",
        "cutoutTemplate",
        "cutoutDifference",
    ]
    _Model = GenericAlert

    @classmethod
    @functools.lru_cache(1)
    def _exclude_from_extra_fields(cls) -> Set[str]:
        """Returns a set of fields that should not be present in `extra_fields` for `GenericAlert`"""
        ignore = {
            mapper.origin
            for mapper in cls._mapping.values()
            if mapper.origin is not None
        }
        ignore.update(cls._ignore_in_extra_fields)
        return ignore

    @classmethod
    @abc.abstractmethod
    def _extract_stamps(cls, message: dict) -> dict:
        """Keys are `science`, `template` and `difference`. Values are of type `byte` or `None`"""
        return {
            "science": None,
            "template": None,
            "difference": None,
        }

    @classmethod
    def parse_message(cls, message: dict) -> GenericAlert:
        """Create a `GenericAlert` from the message"""
        generic = {name: mapper(message) for name, mapper in cls._mapping.items()}

        stamps = cls._extract_stamps(message)
        extra_fields = {
            k: v
            for k, v in message.items()
            if k not in cls._exclude_from_extra_fields()
        }
        return cls._Model(**generic, stamps=stamps, extra_fields=extra_fields)

    @classmethod
    @abc.abstractmethod
    def can_parse(cls, message: dict) -> bool:
        """Whether the message can be parsed"""

    @classmethod
    def get_source(cls) -> str:
        """Name of the parser source"""
        return cls._source
