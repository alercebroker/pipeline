from abc import ABC, abstractmethod
import logging
from typing import Union


class GenericProducer(ABC):
    """Generic Producer for Alert Processing Framework."""

    def __init__(self, config=None):
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config
        self._key_field = None

    @property
    def key_field(self):
        return self._key_field

    def set_key_field(self, key):
        """Set key used when indexing produced messages."""
        self._key_field = key

    @abstractmethod
    def produce(self, message=None, **kwargs):
        """Send a message after processing.

        Parameters
        ----------
        message : dict-like
            Message to be sended.
        """
        pass
