from typing import List
from .generic import GenericAlert, SurveyParser
from ..parsers import ATLASParser, ZTFParser, LSSTParser


class ParserSelector:
    def __init__(self):
        self.parsers = set()

    def __repr__(self):
        return str(self.parsers)

    def register_parser(self, parser: SurveyParser) -> None:
        """Add a parser to the selector"""
        self.parsers.add(parser)

    def remove_parser(self, parser: SurveyParser) -> None:
        """Remove a parser from the selector."""
        try:
            self.parsers.remove(parser)
        except KeyError:  # the parser wasn't present
            pass

    def _parse(self, message: dict) -> GenericAlert:
        """Parse the message into a `GenericAlert`.

        It will iterate over the parsers and the first to be able to parse the message will create the alert.
        """
        for parser in self.parsers:
            if parser.can_parse(message):
                return parser.parse_message(message)
        raise ValueError("Can't parse message")

    def parse(self, messages: List[dict]) -> List[GenericAlert]:
        """Parse all messages into `GenericAlert`s"""
        return [self._parse(message) for message in messages]


class ALeRCEParser(ParserSelector):
    def __init__(self):
        super().__init__()
        self.parsers = {ATLASParser, ZTFParser, LSSTParser}


class ELAsTiCCParser(ParserSelector):
    def __init__(self):
        super().__init__()
        self.parsers = {LSSTParser}
