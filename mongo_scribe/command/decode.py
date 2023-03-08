from json import loads

from .commands import *
from .exceptions import WrongFormatCommandException


def validate(message: dict) -> dict:
    """Checks if a dictionary has a valid command format. Returns the dictionary if valid.

    It raises MisformattedCommand if the JSON string isn't a valid command.
    """
    if any(
        required not in message for required in ["type", "data", "collection"]
    ):
        raise WrongFormatCommandException

    if "criteria" not in message:
        message["criteria"] = {}

    if "options" not in message:
        message["options"] = {}

    return message


def decode_message(encoded_message: str):
    """
    Transforms a JSON string into a Python dictionary.
    """
    decoded = loads(encoded_message)
    valid_message = validate(decoded)

    return valid_message


def db_command_factory(msg: str) -> DbCommand:
    """
    Returns a DbCommand instance based on a JSON stringified.
    Raises MisformattedCommand if the JSON string is not a valid command.
    """
    decoded_message = decode_message(msg)
    msg_type = decoded_message.pop("type")

    if msg_type == InsertDbCommand.type:
        return InsertDbCommand(**decoded_message)
    if msg_type == UpdateDbCommand.type:
        return UpdateDbCommand(**decoded_message)
    if msg_type == UpdateProbabilitiesDbCommand.type:
        return UpdateProbabilitiesDbCommand(**decoded_message)
    raise ValueError(f"Unrecognized command type {msg_type}")
