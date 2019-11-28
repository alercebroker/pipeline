from abc import abstractmethod,ABCMeta

class GenericProducer():
    """Generic Producer for Alert Processing Framework."""
    __metaclass__ = ABCMeta

    def produce(self,message = None):
        """Send a message after processing.

        Parameters
        ----------
        message : dict-like
            Message to be sended.
        """
        pass
