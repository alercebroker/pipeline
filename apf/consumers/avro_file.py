from apf.consumers.generic import GenericConsumer

import fastavro
import os
import glob


class AVROFileConsumer(GenericConsumer):
    """Consume from a AVRO Files Directory.

    **Example:**

    .. code-block:: python

        #settings.py
        CONSUMER_CONFIG = { ...
            "DIRECTORY_PATH": "path/to/avro/directory"
        }

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

        if "consume.messages" in self.config:
            num_messages = self.config["consume.messages"]
        elif "NUM_MESSAGES" in self.config:
            num_messages = self.config["NUM_MESSAGES"]
        else:
            num_messages = 1

        msgs = []
        for file in files:
            self.logger.debug(f"Reading File: {file}")
            with open(file, "rb") as f:
                avro_reader = fastavro.reader(f)
                data = avro_reader.next()
            if num_messages == 1:
                yield data
            else:
                msgs.append(data)
                if len(msgs) == num_messages:
                    return_msgs = msgs.copy()
                    msgs = []
                    yield return_msgs


class AVROInfiniteConsumer(GenericConsumer):
    """Consume from a Infinite AVRO Files Directory.

    **Example:**

    .. code-block:: python

        #settings.py
        CONSUMER_CONFIG = { ...
            "DIRECTORY_PATH": "path/to/avro/directory"
        }

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

        if "consume.messages" in self.config:
            num_messages = self.config["consume.messages"]
        elif "NUM_MESSAGES" in self.config:
            num_messages = self.config["NUM_MESSAGES"]
        else:
            num_messages = 1

        msgs = []
        index = 0

        while True:
            file = files[index % len(files)]
            self.logger.debug(f"Reading File: {file}")
            with open(file, "rb") as f:
                avro_reader = fastavro.reader(f)
                for data in avro_reader:
                    if num_messages == 1:
                        yield data
                    else:
                        msgs.append(data)
                        if len(msgs) == num_messages:
                            return_msgs = msgs.copy()
                            msgs = []
                            yield return_msgs
