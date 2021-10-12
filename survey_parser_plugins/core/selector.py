from typing import List
from survey_parser_plugins.core.generic import SurveyParser
from survey_parser_plugins.parsers import ATLASParser, ZTFParser


class ParserSelector:
    def __init__(self, extra_fields=False):
        self.parsers = {}
        self.extra_fields = extra_fields

    def __repr__(self):
        return str(self.parsers)

    def register_parser(self, parser: SurveyParser) -> None:
        if parser.get_source() not in self.parsers.keys():
            self.parsers.update({parser.get_source(): parser})

    def remove_parser(self, parser: SurveyParser) -> None:
        if parser.get_source() in self.parsers.keys():
            del self.parsers[parser.get_source()]

    def _parse(self, message: dict) -> dict:
        if 'publisher' in message.keys():
            if "ZTF" in message["publisher"]:
                return self.parsers["ZTF"].parse_message(message, extra_fields=self.extra_fields)
            elif "ATLAS" in message["publisher"]:
                return self.parsers["ATLAS"].parse_message(message, extra_fields=self.extra_fields)
            else:
                raise Exception("This message hasn't a parser implemented")
        else:
            raise Exception("This message hasn't a publisher key in the message")

    def parse(self, messages: List[dict]) -> List[dict]:
        return list(map(self._parse, messages))


class ALeRCEParser(ParserSelector):
    def __init__(self, extra_fields=False):
        super().__init__(extra_fields=extra_fields)
        self.parsers = {
            "ATLAS": ATLASParser,
            "ZTF": ZTFParser,
        }
