import os
from apf.producers import KafkaProducer
from db_plugins.db.sql._connection_pipeline import PsqlDatabase


class RawKafkaProducer(KafkaProducer):
    """Producer that prevents serialization of the message"""

    def __init__(self, config: dict):
        warn = "SCHEMA" in config
        config.setdefault(
            "SCHEMA", {"name": "dummy", "type": "record", "fields": []}
        )  # Dummy schema
        super().__init__(config)
        if warn:
            self.logger.warning(
                f"SCHEMA in PRODUCER_CONFIG defined but is ignored by {self.__class__.__name__}"
            )

    def _serialize_message(self, message):
        return message


# Set these paths as needed
INPUT_TOPIC = "lsst"
psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

AVRO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../tests/integration/data/avro_messages",
)
SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../schemas/surveys/lsst/v7_4_alert.avsc",
)


def produce_avro_messages_to_kafka():
    """
    Produce all Avro files in AVRO_DIR to the given Kafka topic as raw bytes using RawKafkaProducer.
    Each file is sent as a single message, preserving the original bytes.
    """
    producer = RawKafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS},
            "TOPIC": INPUT_TOPIC,
            "SCHEMA_PATH": SCHEMA_PATH,
        }
    )
    for filename in sorted(os.listdir(AVRO_DIR)):
        if filename.endswith(".avro"):
            with open(os.path.join(AVRO_DIR, filename), "rb") as f:
                data = f.read()
                producer.produce(data)
    producer.producer.flush()
    print(
        f"Produced Avro messages from {AVRO_DIR} to topic '{INPUT_TOPIC}' on {KAFKA_BOOTSTRAP_SERVERS}"
    )


if __name__ == "__main__":
    psql_db = PsqlDatabase(psql_config)
    psql_db.create_db()
    produce_avro_messages_to_kafka()
