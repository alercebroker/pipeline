from typing import List
from .generic import GenericAlert, SurveyParser
from ..parsers import ATLASParser, ZTFParser


class ParserSelector:
    def __init__(self, alerce_id=True):
        self.parsers = set()
        self.alerce_id = alerce_id

    def __repr__(self):
        return str(self.parsers)

    def register_parser(self, parser: SurveyParser) -> None:
        """
        Add a parser to Selector.

        :param parser: SurveyParser implementation.
        :return:
        """
        if parser not in self.parsers:
            self.parsers.add(parser)

    def remove_parser(self, parser: SurveyParser) -> None:
        """
        Remove a parser of Selector.

        :param parser: SurveyParser implementation.
        :return:
        """
        if parser in self.parsers:
            self.parsers.remove(parser)

    def _parse(self, message: dict) -> GenericAlert:
        """
        Call of parse for each parser. This works with any source. Iterate for each parser and the parser that can
        parse, it parse the message.

        :param message: Message from a survey stream.
        :return: GenericAlert. Parsed messaged.
        """
        for parser in self.parsers:
            if parser.can_parse(message):
                parsed = parser.parse_message(message)
                return parsed
        raise Exception("This message can't be parsed")

    def parse(self, messages: List[dict]) -> List[GenericAlert]:
        """
        Apply for each message the parse function.

        :param messages: List of messages from survey stream.
        :return:
        """
        return list(map(self._parse, messages))


class ALeRCEParser(ParserSelector):
    def __init__(self):
        super().__init__()
        self.parsers = {ATLASParser, ZTFParser}
