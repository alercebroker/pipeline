from apf.producers.generic import GenericProducer

class TestProducer(GenericProducer):

    def __init__(self, config):
        super().__init__(config=config)
        self.pre_produce_message = []

    def produce(self, message=None, **kwargs):
        """Produce Message of dictionary list and save it inside a variable to access it afterwards"""
        self.pre_produce_message.append(message)
        return