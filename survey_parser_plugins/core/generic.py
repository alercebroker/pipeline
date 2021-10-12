import abc


class SurveyParser(abc.ABC):
    _source = None

    @abc.abstractmethod
    def parse_message(self, message: dict):
        """
        :param message: A single message from an astronomical survey. Typically this corresponds a dict.

        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass

    @abc.abstractmethod
    def get_source(self):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass
