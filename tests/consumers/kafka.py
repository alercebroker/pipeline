from .core import GenericConsumerTest
from apf.consumers import KafkaConsumer

class KafkaConsumer(GenericConsumerTest,unittest.TestCase):
    component = KafkaConsumer
