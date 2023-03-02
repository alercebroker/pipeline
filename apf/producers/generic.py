from abc import ABC, abstractmethod
import logging


class GenericProducer(ABC):
    """Generic Producer for Alert Processing Framework."""

    def __init__(self, config=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config
        self._producer_key = None

    @property
    def producer_key(self):
        return self._producer_key

    @producer_key.setter
    def producer_key(self, key: str):
        self._producer_key = key

    @abstractmethod
    def produce(self, message=None, **kwargs):
        """Send a message after processing.

        Parameters
        ----------
        message : dict-like
            Message to be sended.
        """
        pass
