from ..ztf.parser import ZTFParser
from .parser_interface import ParserInterface


def select_parser(strategy: str | None) -> ParserInterface:
    strategy = strategy.lower() if strategy else None
    match strategy:
        case "ztf":
            return ZTFParser()
        case _:
            raise Exception("Invalid 'SURVEY_STRATEGY', must be one of:['ztf']")
