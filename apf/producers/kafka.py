from apf.producers.generic import GenericProducer
from confluent_kafka import Producer

class KafkaProducer(GenericProducer):
    def __init__(self,config,topic):
        super().__init__()
        self.consumer = Producer(config)


    def consume(self):
        return "Test"
