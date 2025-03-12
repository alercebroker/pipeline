from ingestion_step.parser.core.parser_interface import ParserInterface
from ingestion_step.parser.ztf.parser import ZTFParser


def select_parser(strategy: str | None) -> ParserInterface:
    strategy = strategy.lower() if strategy else None
    match strategy:
        case "ztf":
            return ZTFParser()
        case _:
            raise Exception("Invalid 'SURVEY_STRATEGY', must be one of:['ztf']")
