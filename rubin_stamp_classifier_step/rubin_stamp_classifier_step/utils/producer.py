from apf.producers import GenericProducer


class LoggerProducer(GenericProducer):
    """A producer that logs messages instead of sending them to a message broker."""

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
