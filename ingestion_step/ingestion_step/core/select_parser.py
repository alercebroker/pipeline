from typing import Type, cast

from ingestion_step.core.exceptions import SelectorException
from ingestion_step.core.strategy import ParsedData, StrategyInterface
from ingestion_step.lsst.strategy import LsstStrategy
from ingestion_step.ztf.strategy import ZtfStrategy

strategy_registry = {"ztf": ZtfStrategy, "lsst": LsstStrategy}


def select_parser(strategy: str) -> Type[StrategyInterface[ParsedData]]:
    """
    Selects the appropriate Parser for the given strategy/survey.

    Returns an object that implements ParserInterface for the given
    strategy.

    `strategy` must be one of (case-insensitive):
        - 'ztf'
        - 'lsst'
    Otherwise raises `SelectorException`.
    """
    strategy = strategy.lower()
    try:
        # Casting the Strategy into its more generic base type (the implementation)
        # shouldn't matter to the step.
        return cast(
            type[StrategyInterface[ParsedData]], strategy_registry[strategy]
        )
    except KeyError:
        raise SelectorException(
            f"Invalid 'SURVEY_STRATEGY', must be one of: {strategy_registry.keys()}"
        )
