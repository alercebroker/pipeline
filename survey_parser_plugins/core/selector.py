from typing import List
from .generic import GenericAlert, SurveyParser
from ..parsers import ATLASParser, ZTFParser
from .id_generator import id_generator


class ParserSelector:
    def __init__(self, alerce_id=True):
        self.parsers = set()
        self.alerce_id = alerce_id

    def __repr__(self):
        return str(self.parsers)

    @classmethod
    def add_alerce_id(cls, alert: GenericAlert) -> GenericAlert:
        """
        Compute alerce_id and put on generic alert.

        :param alert: GenericAlert of ALeRCE.
        :return: GenericAlert with alerce id.
        """
        alert.aid = id_generator(alert.ra, alert.dec)
        return alert

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
                if self.alerce_id:
                    self.add_alerce_id(parsed)
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
    def __init__(self, alerce_id=True):
        super().__init__(alerce_id=alerce_id)
        self.parsers = {
            ATLASParser,
            ZTFParser
        }
