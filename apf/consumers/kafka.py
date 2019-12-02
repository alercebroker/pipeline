from apf.consumers.generic import GenericConsumer
from confluent_kafka import Consumer

import fastavro
import io

class KafkaConsumer(GenericConsumer):
    def __init__(self,config):
        super().__init__(config)
        #Disable auto commit
        # config["enable.auto.commit"] = False
        # if "max.poll.interval.ms" not in config:
            # config["max.poll.interval.ms"] = 10*60*1000

        if "auto.offset.reset" not in config:
            config["auto.offset.reset"] = "smallest"
        #Creating consumer
        self.init = False

    def _deserialize_message(self,message):
        self.logger.debug(len(message))
        bytes_io = io.BytesIO(message.value())
        reader = fastavro.reader(bytes_io)
        data = reader.next()
        return data

    def _init_consumer(self):
        if not self.init:
            self.consumer = Consumer(self.config["PARAMS"])
            self.logger.info(f'Subscribing to {self.config["TOPICS"]}')
            self.consumer.subscribe(self.config["TOPICS"])

    def consume(self):
        self._init_consumer()
        message = None
        while message is None:
            #Get 1 message with a 60 sec timeout
            messages = self.consumer.consume(num_messages=1,timeout=60)
        data = self._deserialize_message(message)
        yield data

    def commit(self,message):
        pass
        # self.consumer.commit(message)
