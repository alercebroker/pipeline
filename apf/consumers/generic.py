from abc import abstractmethod,ABCMeta

class GenericConsumer():
    __metaclass__ = ABCMeta

    """Generic Consumer for Alert Processing Framework."""
    @abstractmethod
    def consume(self):
        """Get a message from a data source

        Returns
        -------
        dict
            Dictionary like message of an alert.
        """
        pass

    def commit(self,msj):
        """Post consume processing.
        Can be a postgresql, kafka, commit or a custom function to run after an alert is processed.

        Parameters
        ----------
        msj : dict-like
            Message consumed.
        """
        pass
