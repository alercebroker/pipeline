from abc import ABC, abstractmethod
from typing import Any

from .parsed_data import ParsedData


class ParserInterface(ABC):
    @abstractmethod
    def parse(self, messages: list[dict[str, Any]]) -> ParsedData:
        pass
