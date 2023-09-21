from json import loads
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

def decode_message():
    # TODO
    pass

def command_factory():
    # TODO
    pass
