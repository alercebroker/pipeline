from typing import List
from survey_parser_plugins.core.generic import SurveyParser
from survey_parser_plugins.parsers import ATLASParser, ZTFParser


class ParserSelector:
    def __init__(self):
        self.parsers = {}

    def __repr__(self):
        return str(self.parsers)

    def register_parser(self, parser: SurveyParser) -> None:
        if parser.get_source() not in self.parsers.keys():
            self.parsers.update({parser.get_source(): parser})

    def delete_parser(self, parser: SurveyParser) -> None:
        if parser.get_source() in self.parsers.keys():
            del self.parsers[parser.get_source()]

    def _parse(self, message: dict):
        if 'publisher' in message.keys() and "ZTF" in message['publisher']:
            return self.parsers["ZTF"].parse_message(message)
        else:
            raise Exception("This message haven't a parser")

    def parse(self, messages: List[dict]):
        return list(map(self._parse, messages))


class ALeRCEParser(ParserSelector):
    def __init__(self):
        super().__init__()
        self.parsers = {
            "ATLAS": ATLASParser,
            "ZTF": ZTFParser,
        }
