from abc import ABC, abstractmethod
import logging
from typing import Generator, Union


class GenericConsumer(ABC):
    """Generic Consumer for Alert Processing Framework.

    Parameters are passed through *config* as a :py:class:`dict` of params.
    """

    def __init__(self, config=None):
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config

    @abstractmethod
    def consume(self) -> Generator[Union[list, dict], None, None]:
        """Get a message from a data source

        Yields
        ------
        dict
            Dictionary like message of an alert.
        """
        yield {}

    def commit(self):
        """Post consume processing.
        Can be a postgresql, kafka, commit or a custom function to run after an alert is processed.

        The commited value has to be stored as a class attribute in consume to be accessed. i.e.

        .. code-block:: python

            def consume(self):
                self.message = get_message()

            def commit(self):
                commit_logic(self.message)

        """
        pass
