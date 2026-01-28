import logging
from abc import ABC, abstractmethod
from typing import Any, Callable


class GenericProducer(ABC):
    """Generic Producer for Alert Processing Framework."""

    def __init__(self, config=None):
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config
        self._key_field = None
        self._key_function: Callable[[dict[str, Any]], str] | None = None

    @property
    def key_field(self):
        return self._key_field

    @property
    def key_function(self):
        return self._key_function

    def set_key_function(self, function: Callable[[dict[str, Any]], str]):
        """Set function to compute a key based on the message."""
        self._key_function = function

    def set_key_field(self, key):
        """Set key used when indexing produced messages."""
        self._key_field = key

    @abstractmethod
    def produce(self, message=None, **kwargs):
        """Send a message after processing.

        Parameters
        ----------
        message : dict-like
            Message to be sent.
        """
        pass
