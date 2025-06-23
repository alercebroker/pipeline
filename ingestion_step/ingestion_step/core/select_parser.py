from ingestion_step.core.exceptions import SelectorException

from ..ztf.parser import ZTFParser
from .parser_interface import ParserInterface


def select_parser(strategy: str | None) -> ParserInterface:
    """
    Selects the appropriate Parser for the given strategy/survey.

    Returns an object that implements ParserInterface for the given
    strategy.

    `strategy` must be one of (case-insensitive):
        - 'ztf'
    Otherwise raises `SelectorException`.
    """
    strategy = strategy.lower() if strategy else None

    match strategy:
        case "ztf":
            return ZTFParser()
        case _:
            raise SelectorException(
                "Invalid 'SURVEY_STRATEGY', must be one of: ['ztf']"
            )
