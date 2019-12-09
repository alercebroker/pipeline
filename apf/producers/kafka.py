from apf.producers.generic import GenericProducer
from confluent_kafka import Producer

import io
import fastavro


class KafkaProducer(GenericProducer):
    """Kafka Single Topic Producer.

    Parameters
    ----------
    PARAMS: dict
        Parameters passed to :class:`confluent_kafka.Producer`

        The required parameters are:

        - *bootstrap.servers*: comma separated <host:port> :py:class:`str` to brokers.

    TOPIC: string
        Kafka output topic.

    SCHEMA: dict
        AVRO Output Schema `(AVRO Schema Definition) <https://avro.apache.org/docs/current/gettingstartedpython.html#Defining+a+schema>`_

    """
    def __init__(self,config):
        super().__init__(config=config)
        self.producer = Producer(self.config["PARAMS"])
        self.topic = self.config["TOPIC"]
        self.schema = self.config["SCHEMA"]

    def produce(self,message=None):
        """Produce Message to a topic.
        """
        schema = fastavro.parse_schema(self.schema)
        out = io.BytesIO()
        fastavro.writer(out, schema, [message])
        avro_message = out.getvalue()
        self.producer.produce(self.topic,avro_message)
        self.producer.flush()
