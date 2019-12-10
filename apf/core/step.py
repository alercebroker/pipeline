from abc import abstractmethod

from apf.consumers import GenericConsumer

import logging

class GenericStep():
    """Generic Step for apf.

    Parameters
    ----------
    consumer : :class:`GenericConsumer`
        An object of type GenericConsumer.
    level : logging.level
        Logging level, has to be a logging.LEVEL constant.
    **step_args : dict
        Additional parameters for the step.
    """
    def __init__(self,consumer = None, level = logging.INFO,config=None, **step_args):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config
        self.consumer = GenericConsumer() if consumer is None else consumer

    @abstractmethod
    def execute(self, message):
        """Execute the logic of the step. This method has to be implemented by
        the instanced class.

        Parameters
        ----------
        message : dict
            Dict-like message to be processed.
        """
        pass

    def start(self):
        """Start running the step.
        """
        for self.message in self.consumer.consume():
            self.execute(self.message)
            self.consumer.commit()
