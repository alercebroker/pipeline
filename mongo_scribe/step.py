import logging
from apf.core.step import GenericStep
from .command.decode import db_command_factory
from .db.executor import ScribeCommandExecutor


class MongoScribe(GenericStep):
    """MongoScribe Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    """

    def __init__(self, consumer=None, config=None, **step_args):
        super().__init__(consumer, config=config, **step_args)
        self.db_client = ScribeCommandExecutor(config["DB_CONFIG"])

    def execute(self, messages):
        """
        Transforms a batch of messages from a topic into Scribe
        DB Commands and executes them when they're valid.
        NOTE: WE'RE ASSUMING THAT EVERY MESSAGE FROM THE BATCH GOES INTO THE SAME COLLECTION
        """
        logging.info("Processing messages...")
        valid_commands, n_invalid_commands = [], 0
        for message in messages:
            try:
                new_command = db_command_factory(message["payload"])
                valid_commands.append(new_command)
            except Exception as exc:
                logging.error(f"Error processing message: {exc}")
                n_invalid_commands += 1

        logging.info(
            f"Processed {len(valid_commands)} messages successfully. Found {n_invalid_commands} invalid messages."
        )

        if len(valid_commands) > 0:
            logging.info("Writing commands into database")
            # collection = valid_commands[0].collection
            self.db_client.bulk_execute(valid_commands)

        return []
