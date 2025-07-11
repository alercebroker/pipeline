from abc import ABC, abstractmethod
from typing import Generic, TypedDict, TypeVar

from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.types import Message


class ParsedData(TypedDict):
    """
    Dictionary of pandas `DataFrame`.

    The dictionary contains data parsed from the kafka messages associated
    for example for ztf: objects, detections, non_detections and forced_photometries.
    """


T = TypeVar("T", bound=ParsedData)


class StrategyInterface(ABC, Generic[T]):
    """
    Minimal interface for a common survey strategy.

    An implementation of this Interface is returned by the `strategy_selector` to be
    used by the steop to parse the incoming messages.
    """

    @classmethod
    @abstractmethod
    def parse(cls, messages: list[Message]) -> T:
        """
        Parses a list of messages into a `ParsedData` dict of pandas DataFrames.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def insert_into_db(cls, driver: PsqlDatabase, parsed_data: T):
        """
        Insert data obtained from parse method into the given db.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def serialize(cls, parsed_data: T) -> list[Message]:
        """
        Serializes the data to a format ready to send to Kafka.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_key(cls) -> str:
        raise NotImplementedError
