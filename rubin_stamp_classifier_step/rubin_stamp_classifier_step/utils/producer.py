import fastavro.schema
from apf.producers import GenericProducer


class LoggerProducer(GenericProducer):
    """A producer that logs messages instead of sending them to a message broker.

    Give it the output SCHEMA_PATH so the step trims messages to the output
    fields, as it does for a Kafka producer.
    """

    def __init__(self, config=None):
        super().__init__(config)
        if config and config.get("SCHEMA_PATH"):
            self.schema = fastavro.schema.load_schema(config["SCHEMA_PATH"])

    def produce(self, message=None, **kwargs):
        """Log the message instead of producing it."""
        if message is None:
            self.logger.warning("No message provided to LoggerProducer.")
            return

        self.logger.info(f"Producing message: {message}")
        if kwargs:
            self.logger.debug(
                f"Additional kwargs: {kwargs}"
            )  # Log additional keyword arguments if any
