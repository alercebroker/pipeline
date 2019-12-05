from apf.consumers.generic import GenericConsumer
from confluent_kafka import Consumer

import fastavro
import io

class KafkaConsumer(GenericConsumer):
    """Consume from a Kafka Topic

    Parameters
    ----------
    TOPICS: :py:class:`list`
            List of topics to consume.

    PARAMS: dict
        Parameters passed to :class:`confluent_kafka.Consumer`

        The required parameters are:

        - *bootstrap.servers*: comma separated <host:port> :py:class:`str` to brokers.
        - *group.id*: :py:class:`str` with consumer group name.

    """
    def __init__(self,config):
        super().__init__(config)
        # Disable auto commit
        self.config["PARAMS"]["enable.auto.commit"] = False
        if "max.poll.interval.ms" not in config["PARAMS"]:
            self.config["PARAMS"]["max.poll.interval.ms"] = 10*60*1000

        if "auto.offset.reset" not in self.config:
            self.config["PARAMS"]["auto.offset.reset"] = "beginning"
        #Creating consumer
        self.consumer = Consumer(self.config["PARAMS"])
        self.logger.info(f'Subscribing to {self.config["TOPICS"]}')
        self.consumer.subscribe(self.config["TOPICS"])

    def __del__(self):
        self.logger.info("Shutting down Consumer")
        self.consumer.close()

    def _deserialize_message(self,message):
        bytes_io = io.BytesIO(message.value())
        reader = fastavro.reader(bytes_io)
        data = reader.next()
        return data

    def consume(self):
        message = None
        while True:
            while message is None:
                #Get 1 message with a 60 sec timeout
                message = self.consumer.poll(timeout=60)

            if message.error():
                raise Exception("Error in kafka stream: {message.error()}")

            self.message = message

            data = self._deserialize_message(message)
            message = None
            yield data


    def commit(self):
        self.consumer.commit(self.message)
