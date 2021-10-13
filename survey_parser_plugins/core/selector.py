from typing import List
from survey_parser_plugins.core.generic import SurveyParser
from survey_parser_plugins.parsers import ATLASParser, ZTFParser
from survey_parser_plugins.core.id_generator import id_generator


class ParserSelector:
    def __init__(self, extra_fields=False, alerce_id=True):
        self.parsers = set()
        self.extra_fields = extra_fields
        self.alerce_id = alerce_id

    def __repr__(self):
        return str(self.parsers)

    def add_alerce_id(self, message: dict) -> dict:
        message["alerce_id"] = id_generator(message["ra"], message["dec"])
        return message

    def register_parser(self, parser: SurveyParser) -> None:
        if parser not in self.parsers:
            self.parsers.add(parser)

    def remove_parser(self, parser: SurveyParser) -> None:
        if parser in self.parsers:
            self.parsers.remove(parser)

    def _parse(self, message: dict) -> dict:
        for parser in self.parsers:
            if parser.can_parse(message):
                parsed = parser.parse_message(message)
                if self.alerce_id:
                    self.add_alerce_id(parsed)
                return parsed
        else:
            raise Exception("This message can't be parsed")

    def parse(self, messages: List[dict]) -> List[dict]:
        return list(map(self._parse, messages))


class ALeRCEParser(ParserSelector):
    def __init__(self, extra_fields=False):
        super().__init__(extra_fields=extra_fields)
        self.parsers = {
            ATLASParser,
            # DECATParser,
            ZTFParser
        }
