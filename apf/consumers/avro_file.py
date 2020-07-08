from apf.consumers.generic import GenericConsumer
from confluent_kafka import Consumer

import fastavro
import os
import glob


class AVROFileConsumer(GenericConsumer):
    """Consume from a AVRO Files Directory.

    Parameters
    ----------
    DIRECTORY_PATH: path
        AVRO files Directory path location
    """
    def __init__(self, config):
        super().__init__(config)

    def consume(self):
        files = glob.glob(os.path.join(self.config["DIRECTORY_PATH"], "*.avro"))
        files.sort()

        for file in files:
            self.logger.debug(f"Reading File: {file}")
            with open(file, "rb") as f:
                avro_reader = fastavro.reader(f)
                data = avro_reader.next()
            yield data
