import logging

from apf.core.step import GenericStep

from .sql.command.decode import command_factory
from .sql.db.executor import SQLCommandExecutor


class SqlScribe(GenericStep):
    """SqlScribe Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    """

    def __init__(self, consumer=None, config=None, **step_args):
        super().__init__(consumer, config=config, **step_args)

        self.db_client = SQLCommandExecutor(config["PSQL_CONFIG"])

    def execute(self, messages):
        """
        Transforms a batch of messages from a topic into Scribe
        DB Commands and executes them when they're valid.
        """
        logging.info("Processing messages...")
        valid_commands, n_invalid_commands = [], 0
        for message in messages:
            try:
                new_command = command_factory(message["payload"])
                valid_commands.append(new_command)
            except ValueError as e:
                self.logger.debug(e)
                n_invalid_commands += 1

        logging.info(
            f"Processed {len(valid_commands)} messages successfully. Found {n_invalid_commands} invalid messages."
        )

        if len(valid_commands) > 0:
            logging.info("Writing commands into database")
            self.db_client.bulk_execute(valid_commands)

        return []
