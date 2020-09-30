from abc import ABCMeta
import logging


class GenericProducer():
    """Generic Producer for Alert Processing Framework."""
    __metaclass__ = ABCMeta

    def __init__(self, config=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config

    def produce(self, message=None, **kwargs):
        """Send a message after processing.

        Parameters
        ----------
        message : dict-like
            Message to be sended.
        """
        pass
