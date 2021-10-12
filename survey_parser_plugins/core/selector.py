from typing import List
from survey_parser_plugins.core.generic import SurveyParser
from survey_parser_plugins.parsers import ATLASParser, DECATParser, ZTFParser


class ParserSelector:
    def __init__(self, extra_fields=False):
        self.parsers = set()
        self.extra_fields = extra_fields

    def __repr__(self):
        return str(self.parsers)

    def register_parser(self, parser: SurveyParser) -> None:
        if parser not in self.parsers:
            self.parsers.add(parser)

    def remove_parser(self, parser: SurveyParser) -> None:
        if parser in self.parsers:
            self.parsers.remove(parser)

    def _parse(self, message: dict) -> dict:
        for parser in self.parsers:
            if parser.can_parse(message):
                return parser.parse_message(message)
        else:
            print(message.keys())
            raise Exception("This message can't be parsed")

    def parse(self, messages: List[dict]) -> List[dict]:
        return list(map(self._parse, messages))


class ALeRCEParser(ParserSelector):
    def __init__(self, extra_fields=False):
        super().__init__(extra_fields=extra_fields)
        self.parsers = {
            ATLASParser,
            DECATParser,
            ZTFParser
        }
