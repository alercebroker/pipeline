from abc import abstractmethod,ABCMeta
import logging

class GenericConsumer():
    """Generic Consumer for Alert Processing Framework."""
    __metaclass__ = ABCMeta

    def __init__(self,config=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config


    @abstractmethod
    def consume(self):
        """Get a message from a data source

        Returns
        -------
        dict
            Dictionary like message of an alert.
        """
        yield None

    def commit(self,msj):
        """Post consume processing.
        Can be a postgresql, kafka, commit or a custom function to run after an alert is processed.

        Parameters
        ----------
        msj : dict-like
            Message consumed.
        """
        pass
