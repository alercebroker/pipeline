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


def db_command_factory(msg: str) -> Command:
    """
    Returns a DbCommand instance based on a JSON stringified.
    Raises MisformattedCommand if the JSON string is not a valid command.
    """
    decoded_message = decode_message(msg)
    msg_type = decoded_message.pop("type")

    if msg_type == InsertCommand.type:
        return InsertCommand(**decoded_message)
    if msg_type == UpdateCommand.type:
        return UpdateCommand(**decoded_message)
    if msg_type == UpdateProbabilitiesCommand.type:
        return UpdateProbabilitiesCommand(**decoded_message)
    if msg_type == UpdateFeaturesCommand.type:
        return UpdateFeaturesCommand(**decoded_message)
    raise ValueError(f"Unrecognized command type {msg_type}")
