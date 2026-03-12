import pathlib

import pandas as pd

from apf.core.types import Message, MessageBatch
from apf.producers.generic import GenericProducer


class JSONProducer(GenericProducer):
    """JSON Producer

    This producer creates multiple output files (json)
    according to the buffer size, where each file contains
    `buffer_size` elements.

    Every file is created in the `FILE_PATH` directory
    and each output file is named producer_output{i} where `i` is
    a counter for the times the buffer has completed.

    Parameters
    ----------
    FILE_PATH: :py:class:`str`
        Output JSON File Directory.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.buffer = pd.DataFrame()
        self.buffer_size = config.get("buffer_size", 1)
        self.file_counter = 0

    def produce(self, message: Message | MessageBatch, **kwargs):
        """Produce Message to a JSON File."""
        if isinstance(message, dict):
            message = [message]

        if "FILE_PATH" in self.config and self.config["FILE_PATH"]:
            df_message = pd.DataFrame(message)
            self.buffer = pd.concat([self.buffer, df_message])
            output_file = (
                pathlib.Path(self.config["FILE_PATH"])
                / f"producer_output{self.file_counter}.json"
            )
            if len(self.buffer) == self.buffer_size:
                self.logger.info(f"Buffer: {self.buffer}")
                self.buffer.to_json(
                    output_file,
                    orient="records",
                )
                self.file_counter += 1
                self.buffer = []
