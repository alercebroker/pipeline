from apf.consumers.generic import GenericConsumer
from confluent_kafka import Consumer

class KafkaConsumer(GenericConsumer):
    def __init__(self,config,topic):
        super().__init__()
        self.consumer = Consumer(config)


    def consume(self):
        return "Test"
