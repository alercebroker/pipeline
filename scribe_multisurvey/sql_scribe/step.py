import logging

from apf.core.step import GenericStep
from typing import List

from .sql.command.decode import command_factory
from .sql.db.executor import SQLCommandExecutor
import json
from json import loads


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
        self.allowed_commands = config.get("ALLOWED_COMMANDS", None) 
        
    def pre_execute(self, messages: List[dict]) -> dict:
        magstat_messages = {}
        magstat_objects_messages = {}
        other_messages = []
        
        for message in messages:
            step = message.get("step")
            
            if step == "magstat":
                oid = message["payload"]["oid"]
                payload = message["payload"]
                current_counts = (
                    payload.get("n_det", 0),
                    payload.get("n_fphot", 0),
                    payload.get("ndubious", 0)
                )
                if oid not in magstat_messages:
                    magstat_messages[oid] = {
                        "message": message,
                        "counts": current_counts
                    }
                else:
                    if current_counts > magstat_messages[oid]["counts"]:
                        magstat_messages[oid] = {
                            "message": message,
                            "counts": current_counts
                        }
            elif step == "magstat_objects":
                oid = message["payload"]["oid"]
                lastmjd = message["payload"]["lastmjd"]
                
                if oid not in magstat_objects_messages:
                    magstat_objects_messages[oid] = {
                        "message": message,
                        "lastmjd": lastmjd
                    }
                else:
                    if lastmjd > magstat_objects_messages[oid]["lastmjd"]:
                        magstat_objects_messages[oid] = {
                            "message": message,
                            "lastmjd": lastmjd
                        }
            else:
                other_messages.append(message)
        
        # Extract the filtered messages
        filtered_magstat_messages = [item["message"] for item in magstat_messages.values()]
        filtered_magstat_objects_messages = [item["message"] for item in magstat_objects_messages.values()]
        
        # Combine all filtered messages
        messages_filtered = filtered_magstat_messages + filtered_magstat_objects_messages + other_messages
        return messages_filtered

    def execute(self, messages):
        """
        Transforms a batch of messages from a topic into Scribe
        DB Commands and executes them when they're valid.
        """
        logging.info("Processing messages...")
        valid_commands, n_invalid_commands, n_skipped_commands = [], 0, 0

        for message in messages:
            try:
                new_command = command_factory(message["payload"])
                if self.allowed_commands is None or new_command.type in self.allowed_commands:
                    valid_commands.append(new_command)
                else:
                    self.logger.debug(
                        f"Skipping command of type '{new_command.type}' (not in ALLOWED_COMMANDS)"
                    )
                    n_skipped_commands += 1
            except ValueError as e:
                self.logger.debug(e)
                n_invalid_commands += 1

        logging.info(
            f"Processed {len(valid_commands)} messages successfully. "
            f"Found {n_invalid_commands} invalid and {n_skipped_commands} skipped messages."
        )

        if valid_commands:
            logging.info("Writing commands into database")
            self.db_client.bulk_execute(valid_commands)

        return []
