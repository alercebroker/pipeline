from json import loads
from .exceptions import WrongFormatCommandException
from .commands import (
    Command,
    ZTFCorrectionCommand,
    ZTFMagstatCommand,
    LSSTMagstatCommand,
    LSSTFeatureCommand
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
    Get the JSON string and transform it into a Python dictionary.
    """
    # Ensure valid JSON: wrap if missing curly braces
    if not encoded_message.strip().startswith("{"):
        encoded_message = "{" + encoded_message
    if not encoded_message.strip().endswith("}"):
        encoded_message = encoded_message + "}"
    
    decoded = loads(encoded_message)
    
    if (
        "payload" in decoded
        and isinstance(decoded["payload"], dict)
        and "step" in decoded["payload"]
        and "survey" in decoded["payload"]
    ):
        decoded = decoded["payload"]  # unwrap one layer

    valid_message = validate(decoded)

    return valid_message


def command_factory(msg: str) -> Command:
    message = decode_message(msg)
    survey = message.pop("survey")
    step = message.pop("step")

    if survey == "ztf" and step == "correction":
        return ZTFCorrectionCommand(
            payload=message["payload"],
            criteria=message.get("criteria", {}),
            options=message.get("options", {})
        )

    if survey == "ztf" and step == "magstat":
        return ZTFMagstatCommand(
            payload=message["payload"],
            criteria=message.get("criteria", {}),
            options=message.get("options", {})
        ) 

    if survey == "lsst" and step == "magstat":
        return LSSTMagstatCommand(
            payload=message["payload"],
            criteria=message.get("criteria", {}),
            options=message.get("options", {})
        )

    if survey == "lsst" and step == "features":
        return LSSTFeatureCommand(**message)
    
    raise ValueError(f"Unrecognized command type {survey} in table {step}.")

