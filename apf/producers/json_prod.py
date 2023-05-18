from apf.producers.generic import GenericProducer
import json
from pandas import DataFrame, read_json, concat


class JSONProducer(GenericProducer):
    """JSON Producer

    Parameters
    ----------
    FILE_PATH: :py:class:`str`
        Output JSON File Path.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.buffer = DataFrame()
        self.buffer_size = config.get("buffer_size", 1)
        self.file_counter = 0

    def produce(self, message=None, **kwargs):
        """Produce Message to a JSON File."""
        if "FILE_PATH" in self.config and self.config["FILE_PATH"]:
            serialized_message = read_json(
                json.dumps([message]), orient="records", typ="frame"
            )
            self.buffer = concat([self.buffer, serialized_message])
            file_name_list = self.config["FILE_PATH"].split(".")
            file_name = file_name_list[0] + str(self.file_counter)
            if len(self.buffer) == self.buffer_size:
                serialized_message.to_json(
                    file_name + ".json",
                    orient="records",
                    lines=True,
                )
            self.file_counter += 1
