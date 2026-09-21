"""A producer the step can load by import path that records what it is given.

It is a real apf GenericProducer so the framework discovers it when it
drains every producer before committing the consumer offset.
"""
import fastavro.schema
from apf.producers.generic import GenericProducer


class RecordingProducer(GenericProducer):
    def __init__(self, config=None):
        super().__init__(config)
        self.produced = []
        self.flushes = 0
        # Like KafkaProducer: the loaded output schema, when one is configured.
        if config and config.get("SCHEMA_PATH"):
            self.schema = fastavro.schema.load_schema(config["SCHEMA_PATH"])

    def produce(self, message=None, **kwargs):
        self.produced.append((message, kwargs))

    def flush(self):
        self.flushes += 1
