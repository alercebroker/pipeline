from apf.producers.generic import GenericProducer
import json
from pandas import read_json


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
            serialized_message = read_json(
                json.dumps([message]), orient="records", typ="frame"
            )
            serialized_message.to_json(
                self.config["FILE_PATH"], orient="records", lines=True, mode="a"
            )
