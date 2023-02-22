from apf.producers.generic import GenericProducer
import json


class JSONProducer(GenericProducer):
    """JSON Producer

    Parameters
    ----------
    FILE_PATH: :py:class:`str`
        Output JSON File Path.
    """

    def __init__(self, config):
        super().__init__(config=config)

    def produce(self, message=None, **kwargs):
        """Produce Message to a JSON File."""
        if "FILE_PATH" in self.config and self.config["FILE_PATH"]:
            with open(self.config["FILE_PATH"], "a") as outfile:
                json.dump(message, outfile)
