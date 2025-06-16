from json import loads
from .exceptions import WrongFormatCommandException
from .commands import (
    Command,
    ZTFCorrectionCommand,
    ZTFMagstatCommand,
)


def validate(message: dict) -> dict:
    """Checks if a dictionary has a valid command format. Returns the dictionary if valid.

    It raises MisformattedCommand if the JSON string isn't a valid command.
    """
    if any(
        required not in message for required in ["payload", "survey", "step"]
    ):
        raise WrongFormatCommandException

    if "criteria" not in message:
        message["criteria"] = {}

    if "options" not in message:
        message["options"] = {}

    return message


def decode_message(encoded_message: str) -> dict:
    """
    Get the JSON string and transforms it into a Python dictionary.
    """
    decoded = loads(encoded_message)
    valid_message = validate(decoded)
    return valid_message


def command_factory(msg: str) -> Command:
    message = decode_message(msg)
    survey = message.pop("survey")
    step = message.pop("step")

    # here it comes
    if survey == "ztf" and step == "correction":
        return ZTFCorrectionCommand(**message)
    if survey == "ztf" and step == "magstat":
        return ZTFMagstatCommand(**message)
    raise ValueError(f"Unrecognized command type {survey} in table {step}.")
