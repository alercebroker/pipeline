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
    def parse_message(self, message: dict, extra_fields: bool = False) -> dict:
        """
        :param message: A single message from an astronomical survey. Typically this corresponds a dict.
        :param extra_fields: Boolean than indicates if the parser get all remained data on a field called 'extra_fields'

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
