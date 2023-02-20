from json import loads
from mongo_scribe.command.commands import (
    DbCommand,
    InsertDbCommand,
    UpdateDbCommand,
    UpdateProbabilitiesDbCommand,
)
from mongo_scribe.command.exceptions import MisformattedCommandExcepction


def validate(message: dict):
    """Checks if a dictionary has a valid command format. Returns the dictionary if valid, otherwise returns None."""
    if (
        "type" not in message
        or "data" not in message
        or "collection" not in message
    ):
        return None

    if "criteria" not in message:
        message["criteria"] = None

    if "options" not in message:
        message["options"] = None

    if "classifier" not in message["data"]:
        message["data"]["classifier"] = None

    return message


def decode_message(encoded_message: str):
    """
    Transforms a JSON string into a Python dictionary.
    It raises MisformattedCommand if the JSON string isn't a valid command.
    """
    decoded = loads(encoded_message)
    valid_message = validate(decoded)

    if valid_message is None:
        raise MisformattedCommandExcepction

    return valid_message


def db_command_factory(msg: str) -> DbCommand:
    """
    Returns a DbCommand instance based on a JSON stringified.
    Raises MisformattedCommand if the JSON string is not a valid command.
    """
    decoded_message = decode_message(msg)
    msg_type = decoded_message["type"]

    if msg_type == "insert":
        return InsertDbCommand(
            decoded_message["collection"],
            decoded_message["type"],
            decoded_message["criteria"],
            decoded_message["data"],
            decoded_message["options"],
        )

    if msg_type == "update":
        return UpdateDbCommand(
            decoded_message["collection"],
            decoded_message["type"],
            decoded_message["criteria"],
            decoded_message["data"],
            decoded_message["options"],
        )

    if msg_type == "update_probabilities":
        return UpdateProbabilitiesDbCommand(
            decoded_message["collection"],
            decoded_message["type"],
            decoded_message["criteria"],
            decoded_message["data"],
            decoded_message["options"],
        )
